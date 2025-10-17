#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <mutex>
#include <chrono>
#include <cstring>

#include "crow.h"
#include <crow/json.h>

using namespace std;

const double K1 = 1.2;
const double B = 0.75;
const int BLOCK_SIZE = 128;

struct SearchResult {
    int docID;
    double score;
    string snippet;
};
vector<pair<long long, int>> docStoreIndex;

unordered_map<string, tuple<long long, int, int, int>> lexicon;
vector<int> lastDocIDs, docIDSizes, freqSizes;
unordered_map<int, int> docLengths;
int totalDocuments = 0;
double avgDocLength = 0.0;

mutex fileMutex;

// Varbyte decoding
int varbyte_decode(const unsigned char* data, int& offset) {
    int num = 0;
    int shift = 0;
    unsigned char byte;
    
    do {
        byte = data[offset++];
        num |= (byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);
    
    return num;
}

// Tokenize query
vector<string> tokenize(const string& s) {
    vector<string> tokens;
    string token;
    for (char c : s) {
        if (isalnum(c)) {
            token += tolower(c);
        } else if (!token.empty()) {
            tokens.push_back(token);
            token.clear();
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

// Inverted List API
class InvertedList {
private:
    ifstream* invFile;
    string term;
    long long startOffset;
    int startBlock;
    int numPostings;
    
    int currentBlockIdx;
    vector<int> currentDocIDs;
    vector<int> currentFreqs;
    int positionInBlock;
    bool finished;
    
    void decompressBlock(int blockIdx) {
        currentDocIDs.clear();
        currentFreqs.clear();
        
        if (blockIdx >= startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            finished = true;
            return;
        }
        
        long long offset = startOffset;
        for (int i = startBlock; i < blockIdx; i++) {
            offset += sizeof(int) + docIDSizes[i] + sizeof(int) + freqSizes[i];
        }
        
        invFile->seekg(offset);
        
        int docIDSize;
        invFile->read(reinterpret_cast<char*>(&docIDSize), sizeof(int));
        vector<unsigned char> docIDBlock(docIDSize);
        invFile->read(reinterpret_cast<char*>(docIDBlock.data()), docIDSize);
        
        int freqSize;
        invFile->read(reinterpret_cast<char*>(&freqSize), sizeof(int));
        vector<unsigned char> freqBlock(freqSize);
        invFile->read(reinterpret_cast<char*>(freqBlock.data()), freqSize);
        
        int offset_docID = 0;
        while (offset_docID < docIDSize) {
            currentDocIDs.push_back(varbyte_decode(docIDBlock.data(), offset_docID));
        }
        
        for (size_t i = 1; i < currentDocIDs.size(); i++) {
            currentDocIDs[i] += currentDocIDs[i-1];
        }
        
        int offset_freq = 0;
        while (offset_freq < freqSize) {
            currentFreqs.push_back(varbyte_decode(freqBlock.data(), offset_freq));
        }
        
        positionInBlock = 0;
    }
    
public:
    InvertedList(ifstream* file, const string& t) 
        : invFile(file), term(t), finished(false) {
        
        auto it = lexicon.find(term);
        if (it == lexicon.end()) {
            finished = true;
            return;
        }
        
        startOffset = get<0>(it->second);
        startBlock = get<1>(it->second);
        numPostings = get<2>(it->second);
        
        currentBlockIdx = startBlock;
        decompressBlock(currentBlockIdx);
    }
    
    bool nextGEQ(int targetDocID) {
        if (finished) return false;
        
        while (currentBlockIdx < startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            if (lastDocIDs[currentBlockIdx] >= targetDocID) {
                if (currentDocIDs.empty() || positionInBlock >= (int)currentDocIDs.size()) {
                    decompressBlock(currentBlockIdx);
                }
                break;
            }
            currentBlockIdx++;
        }
        
        if (finished) return false;
        
        while (positionInBlock < (int)currentDocIDs.size()) {
            if (currentDocIDs[positionInBlock] >= targetDocID) {
                return true;
            }
            positionInBlock++;
        }
        
        currentBlockIdx++;
        if (currentBlockIdx >= startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            finished = true;
            return false;
        }
        
        decompressBlock(currentBlockIdx);
        return nextGEQ(targetDocID);
    }
    
    bool hasNext() {
        return !finished && positionInBlock < (int)currentDocIDs.size();
    }
    
    int getDocID() {
        if (!hasNext()) return -1;
        return currentDocIDs[positionInBlock];
    }
    
    int getFrequency() {
        if (!hasNext()) return 0;
        return currentFreqs[positionInBlock];
    }
    
    void next() {
        positionInBlock++;
        if (positionInBlock >= (int)currentDocIDs.size()) {
            currentBlockIdx++;
            if (currentBlockIdx < startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
                decompressBlock(currentBlockIdx);
            } else {
                finished = true;
            }
        }
    }
};

// BM25 score
double calculateBM25(int tf, int docLength, int df, int N) {
    double idf = log((N - df + 0.5) / (df + 0.5));
    double tfComponent = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * (docLength / avgDocLength)));
    return idf * tfComponent;
}


bool getDocumentText(int docID, string& text) {
    if (docID < 0 || docID >= (int)docStoreIndex.size()) {
        return false;
    }

    ifstream docStoreFile("index/documents.dat", ios::binary);
    if (!docStoreFile.is_open()) return false;

    long long offset = docStoreIndex[docID].first;
    int length = docStoreIndex[docID].second;

    text.resize(length);
    docStoreFile.seekg(offset);
    docStoreFile.read(&text[0], length);
    docStoreFile.close();
    return true;
}

string generateSnippet(const string& text, const vector<string>& queryTerms, bool forCli) {
    const int SNIPPET_WORDS = 30;
    unordered_set<string> qTerms(queryTerms.begin(), queryTerms.end());

    vector<string> docWords;
    string word;
    for (char c : text) {
        if (isspace(c)) {
            if (!word.empty()) {
                docWords.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty()) docWords.push_back(word);
    
    if (docWords.empty()) return "";

    int bestWindowStart = 0;
    int maxScore = -1;

    for (int i = 0; i <= (int)docWords.size() - SNIPPET_WORDS; ++i) {
        unordered_set<string> foundTerms;
        for (int j = 0; j < SNIPPET_WORDS; ++j) {
            string lowerWord = docWords[i+j];
            transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
            lowerWord.erase(remove_if(lowerWord.begin(), lowerWord.end(), ::iswpunct), lowerWord.end());
            if (qTerms.count(lowerWord)) {
                foundTerms.insert(lowerWord);
            }
        }
        if ((int)foundTerms.size() > maxScore) {
            maxScore = foundTerms.size();
            bestWindowStart = i;
        }
    }
    
    if (maxScore == 0) {
        bestWindowStart = 0;
    }

    stringstream ss;
    if (bestWindowStart > 0) ss << "... ";

    int end = min((int)docWords.size(), bestWindowStart + SNIPPET_WORDS);
    for (int i = bestWindowStart; i < end; ++i) {
        string lowerWord = docWords[i];
        transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
        lowerWord.erase(remove_if(lowerWord.begin(), lowerWord.end(), ::iswpunct), lowerWord.end());

        if (qTerms.count(lowerWord)) {
            if (forCli) ss << "\033[1;31m" << docWords[i] << "\033[0m "; 
            else ss << "\'" << docWords[i] << "\' ";
        } else {
            ss << docWords[i] << " ";
        }
    }
    if (end < (int)docWords.size()) ss << "...";

    return ss.str();
}

// Disjunctive query
vector<pair<int, double>> processDisjunctiveQuery(ifstream* invFile, const vector<string>& queryTerms) {
    unordered_map<int, double> docScores;
    
    for (const auto& term : queryTerms) {
        auto it = lexicon.find(term);
        if (it == lexicon.end()) continue;
        
        int df = get<3>(it->second);
        InvertedList list(invFile, term);
        
        list.nextGEQ(0);
        while (list.hasNext()) {
            int docID = list.getDocID();
            int freq = list.getFrequency();
            int docLen = docLengths.count(docID) ? docLengths[docID] : (int)avgDocLength;
            
            double score = calculateBM25(freq, docLen, df, totalDocuments);
            docScores[docID] += score;
            
            list.next();
        }
    }
    
    vector<pair<int, double>> results;
    for (const auto& pair : docScores) {
        results.push_back({pair.first, pair.second});
    }
    
    sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    
    return results;
}

// Conjunctive query
vector<pair<int, double>> processConjunctiveQuery(ifstream* invFile, const vector<string>& queryTerms) {
    if (queryTerms.empty()) return {};
    
    struct TermInfo {
        string term;
        int df;
        InvertedList* list;
    };
    
    vector<TermInfo> termInfos;
    
    for (const auto& term : queryTerms) {
        auto it = lexicon.find(term);
        if (it == lexicon.end()) {
            for (auto& t : termInfos) delete t.list;
            return {};
        }
        
        int df = get<3>(it->second);
        termInfos.push_back({term, df, new InvertedList(invFile, term)});
    }
    
    sort(termInfos.begin(), termInfos.end(),
         [](const TermInfo& a, const TermInfo& b) { return a.df < b.df; });
    
    vector<InvertedList*> lists;
    vector<int> dfs;
    for (auto& t : termInfos) {
        lists.push_back(t.list);
        dfs.push_back(t.df);
    }

    unordered_map<int, double> docScores;
    
    lists[0]->nextGEQ(0);
    while (lists[0]->hasNext()) {
        int docID = lists[0]->getDocID();
        bool inAll = true;
        vector<int> freqs = {lists[0]->getFrequency()};
        
        for (size_t i = 1; i < lists.size(); i++) {
            lists[i]->nextGEQ(docID);
            if (!lists[i]->hasNext() || lists[i]->getDocID() != docID) {
                inAll = false;
                break;
            }
            freqs.push_back(lists[i]->getFrequency());
        }
        
        if (inAll) {
            int docLen = docLengths.count(docID) ? docLengths[docID] : (int)avgDocLength;
            double totalScore = 0.0;
            for (size_t i = 0; i < lists.size(); i++) {
                totalScore += calculateBM25(freqs[i], docLen, dfs[i], totalDocuments);
            }
            docScores[docID] = totalScore;
        }
        
        lists[0]->next();
    }
    
    for (auto* list : lists) delete list;
    
    vector<pair<int, double>> results;
    for (const auto& p : docScores) results.emplace_back(p.first, p.second);
    
    sort(results.begin(), results.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return results;
}

// Load index
bool loadIndex() {
    ifstream lexFile("index/lexicon.txt");
    if (!lexFile.is_open()) return false;
    
    string line;
    while (getline(lexFile, line)) {
        stringstream ss(line);
        string term;
        long long offset;
        int startBlock, numPostings, df;
        ss >> term >> offset >> startBlock >> numPostings >> df;
        lexicon[term] = make_tuple(offset, startBlock, numPostings, df);
    }
    lexFile.close();
    
    ifstream metaFile("index/metadata.bin", ios::binary);
    if (!metaFile.is_open()) return false;
    
    int numBlocks;
    metaFile.read(reinterpret_cast<char*>(&numBlocks), sizeof(int));
    
    lastDocIDs.resize(numBlocks);
    docIDSizes.resize(numBlocks);
    freqSizes.resize(numBlocks);
    
    metaFile.read(reinterpret_cast<char*>(lastDocIDs.data()), numBlocks * sizeof(int));
    metaFile.read(reinterpret_cast<char*>(docIDSizes.data()), numBlocks * sizeof(int));
    metaFile.read(reinterpret_cast<char*>(freqSizes.data()), numBlocks * sizeof(int));
    metaFile.close();
    
    ifstream docLenFile("index/doc_lengths.txt");
    if (docLenFile.is_open()) {
        while (getline(docLenFile, line)) {
            stringstream ss(line);
            int docID, length;
            ss >> docID >> length;
            docLengths[docID] = length;
            avgDocLength += length;
        }
        totalDocuments = docLengths.size();
        docLenFile.close();
        if (totalDocuments > 0) avgDocLength /= totalDocuments;
    }
    
    ifstream docIdxFile("index/documents.idx", ios::binary);
    if(docIdxFile.is_open()) {
        long long offset;
        int length;
        while(docIdxFile.read(reinterpret_cast<char*>(&offset), sizeof(long long)) && 
              docIdxFile.read(reinterpret_cast<char*>(&length), sizeof(int))) {
            docStoreIndex.push_back({offset, length});
        }
        docIdxFile.close();
    } else {
        return false;
    }
    
    return true;
}

// CLI
void handleCli() {
    cout << "Search engine ready. Type 'quit' to exit.\n";
    cout << "Prefix queries with 'AND:' for conjunctive, 'OR:' for disjunctive (default).\n\n";
    
    ifstream invFile("index/inverted_index.bin", ios::binary);
    if (!invFile.is_open()) {
        cerr << "error opening inverted index\n";
        return;
    }
    
    string line;
    while (true) {
        cout << "Query> ";
        if (!getline(cin, line)) break;
        
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.empty()) continue;
        if (line == "quit" || line == "exit") break;
        
        bool conjunctive = false;
        string query = line;
        
        if (line.size() >= 4 && line.substr(0, 4) == "AND:") {
            conjunctive = true;
            query = line.substr(4);
        } else if (line.size() >= 3 && line.substr(0, 3) == "OR:") {
            query = line.substr(3);
        }
        
        vector<string> queryTerms = tokenize(query);
        if (queryTerms.empty()) continue;
        
        auto start_time = chrono::high_resolution_clock::now();
        
        vector<pair<int, double>> scoredDocs;
        if (conjunctive) {
            scoredDocs = processConjunctiveQuery(&invFile, queryTerms);
        } else {
            scoredDocs = processDisjunctiveQuery(&invFile, queryTerms);
        }
        
        vector<SearchResult> results;
        for(int i = 0; i < min(10, (int)scoredDocs.size()); ++i) {
            string docText;
            string snippet = "Snippet not available.";
            if (getDocumentText(scoredDocs[i].first, docText)) {
                snippet = generateSnippet(docText, queryTerms, true);
            }
            results.push_back({scoredDocs[i].first, scoredDocs[i].second, snippet});
        }

        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_ms = chrono::duration<double, milli>(end_time - start_time).count();
        
        cout << "\nTop " << results.size() << " results:\n";
        for (int i = 0; i < (int)results.size(); i++) {
            cout << (i + 1) << ". DocID: " << results[i].docID 
                 << " (score: " << results[i].score << ")\n";
            cout << "Snippet: " << results[i].snippet << "\n";
        }
        cout << "--------------------------------------------------\n";
        cout << "Total found: " << scoredDocs.size() << " documents\n";
        cout << "Search time: " << elapsed_ms << " ms\n\n";
    }
    
    invFile.close();
}

// Server
void handleServer(int port) {
    crow::SimpleApp app;
    
    CROW_ROUTE(app, "/search")
    ([&](const crow::request& req) {
        auto start_time = chrono::high_resolution_clock::now();
        
        string query = req.url_params.get("q") ? req.url_params.get("q") : "";
        string mode = req.url_params.get("mode") ? req.url_params.get("mode") : "or";
        int limit = req.url_params.get("limit") ? stoi(req.url_params.get("limit")) : 10;
        
        if (query.empty()) {
            return crow::response(400, "{\"error\": \"Missing query parameter 'q'\"}");
        }
        if (mode != "and" && mode != "or") {
            return crow::response(400, "{\"error\": \"Invalid mode. Use 'and' or 'or'\"}");
        }
        limit = max(1, min(100, limit));
        
        vector<string> queryTerms = tokenize(query);
        if (queryTerms.empty()) {
            return crow::response(400, "{\"error\": \"No valid query terms found\"}");
        }
        
        vector<pair<int, double>> scoredDocs;
        
        {
            lock_guard<mutex> lock(fileMutex);
            ifstream invFile("index/inverted_index.bin", ios::binary);
            if (!invFile.is_open()) {
                return crow::response(500, "{\"error\": \"Failed to open inverted index\"}");
            }
            if (mode == "and") {
                scoredDocs = processConjunctiveQuery(&invFile, queryTerms);
            } else {
                scoredDocs = processDisjunctiveQuery(&invFile, queryTerms);
            }
            invFile.close();
        }
        
        vector<SearchResult> results;
        for(int i = 0; i < min(limit, (int)scoredDocs.size()); ++i) {
            string docText;
            string snippet = "Snippet not available.";
            if (getDocumentText(scoredDocs[i].first, docText)) {
                snippet = generateSnippet(docText, queryTerms, false);
            }
            results.push_back({scoredDocs[i].first, scoredDocs[i].second, snippet});
        }
        
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_ms = chrono::duration<double, milli>(end_time - start_time).count();
        
        crow::json::wvalue response;
        response["query"] = query;
        response["total_results"] = (int)scoredDocs.size();
        response["returned_results"] = (int)results.size();
        response["search_time"] = elapsed_ms;
        
        crow::json::wvalue::list result_list;
        for (const auto& res : results) {
            crow::json::wvalue r;
            r["docId"] = res.docID;
            r["score"] = res.score;
            r["snippet"] = res.snippet;
            result_list.push_back(std::move(r));
        }
        response["results"] = std::move(result_list);
        
        return crow::response(response);
    });
    
    app.port(port).multithreaded().run();
}

int main(int argc, char* argv[]) {
    cout << "Loading index..." << endl;
    if (!loadIndex()) {
        cerr << "Error loading index files! Make sure all index files are present in the 'index/' directory." << endl;
        return 1;
    }
    cout << "Index loaded successfully." << endl;
    
    if (argc == 1) {
        handleCli(); 
    } else if (argc >= 2 && strcmp(argv[1], "--server") == 0) {
        int port = 8080; 
        if (argc >= 3) {
            try {
                port = stoi(argv[2]);
            } catch (...) {
                port = 8080;
            }
        }
        handleServer(port);
    } else {
        cerr << "Usage: " << argv[0] << " [--server PORT]\n";
        return 1;
    }
    
    return 0;
}