#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cctype>

using namespace std;

const double K1 = 1.2;
const double B = 0.75;
const int BLOCK_SIZE = 128;

// Global data
unordered_map<string, tuple<long long, int, int, int>> lexicon;
vector<int> lastDocIDs, docIDSizes, freqSizes;
unordered_map<int, int> docLengths;
int totalDocuments = 0;
double avgDocLength = 0.0;

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

// Tokenize
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
    ifstream& invFile;
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
        
        // Calculate file position
        long long offset = startOffset;
        for (int i = startBlock; i < blockIdx; i++) {
            offset += sizeof(int) + docIDSizes[i] + sizeof(int) + freqSizes[i];
        }
        
        invFile.seekg(offset);
        
        // Read docID block
        int docIDSize;
        invFile.read(reinterpret_cast<char*>(&docIDSize), sizeof(int));
        
        vector<unsigned char> docIDBlock(docIDSize);
        invFile.read(reinterpret_cast<char*>(docIDBlock.data()), docIDSize);
        
        // Read freq block
        int freqSize;
        invFile.read(reinterpret_cast<char*>(&freqSize), sizeof(int));
        
        vector<unsigned char> freqBlock(freqSize);
        invFile.read(reinterpret_cast<char*>(freqBlock.data()), freqSize);
        
        // Decode docIDs
        int offset_docID = 0;
        while (offset_docID < docIDSize) {
            currentDocIDs.push_back(varbyte_decode(docIDBlock.data(), offset_docID));
        }
        
        // Convert deltas to absolute
        for (size_t i = 1; i < currentDocIDs.size(); i++) {
            currentDocIDs[i] += currentDocIDs[i-1];
        }
        
        // Decode frequencies
        int offset_freq = 0;
        while (offset_freq < freqSize) {
            currentFreqs.push_back(varbyte_decode(freqBlock.data(), offset_freq));
        }
        
        positionInBlock = 0;
    }
    
public:
    InvertedList(ifstream& file, const string& t) 
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
        
        // Skip blocks using metadata
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
        
        // Find within block
        while (positionInBlock < (int)currentDocIDs.size()) {
            if (currentDocIDs[positionInBlock] >= targetDocID) {
                return true;
            }
            positionInBlock++;
        }
        
        // Move to next block
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

// Disjunctive query
vector<pair<int, double>> processDisjunctiveQuery(ifstream& invFile, const vector<string>& queryTerms) {
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
vector<pair<int, double>> processConjunctiveQuery(ifstream& invFile, const vector<string>& queryTerms) {
    if (queryTerms.empty()) return {};
    
    struct TermInfo {
        string term;
        int df;
        InvertedList* list;
    };
    
    vector<TermInfo> termInfos;
    
    // Step 1: Build InvertedLists and store df (document frequency)
    for (const auto& term : queryTerms) {
        auto it = lexicon.find(term);
        if (it == lexicon.end()) {
            for (auto& t : termInfos) delete t.list;
            return {};
        }
        
        int df = get<3>(it->second); // document frequency from lexicon
        termInfos.push_back({term, df, new InvertedList(invFile, term)});
    }
    
    // Step 2: Sort by ascending df (shortest posting list first)
    sort(termInfos.begin(), termInfos.end(),
         [](const TermInfo& a, const TermInfo& b) { return a.df < b.df; });
    
    // Step 3: Extract pointers and dfs in sorted order
    vector<InvertedList*> lists;
    vector<int> dfs;
    for (auto& t : termInfos) {
        lists.push_back(t.list);
        dfs.push_back(t.df);
    }

    unordered_map<int, double> docScores;
    
    // Step 4: Begin traversal from the *shortest* list
    lists[0]->nextGEQ(0);
    while (lists[0]->hasNext()) {
        int docID = lists[0]->getDocID();
        bool inAll = true;
        vector<int> freqs = {lists[0]->getFrequency()};
        
        // Step 5: Intersect with the rest of the lists
        for (size_t i = 1; i < lists.size(); i++) {
            lists[i]->nextGEQ(docID);
            if (!lists[i]->hasNext() || lists[i]->getDocID() != docID) {
                inAll = false;
                break;
            }
            freqs.push_back(lists[i]->getFrequency());
        }
        
        // Step 6: Score documents that appear in all lists
        if (inAll) {
            int docLen = docLengths.count(docID) ? docLengths[docID] : (int)avgDocLength;
            double totalScore = 0.0;
            for (size_t i = 0; i < lists.size(); i++) {
                totalScore += calculateBM25(freqs[i], docLen, dfs[i], totalDocuments);
            }
            docScores[docID] = totalScore;
        }
        
        lists[0]->next(); // move shortest list forward
    }
    
    // Step 7: Cleanup and sort results
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
            totalDocuments++;
            avgDocLength += length;
        }
        docLenFile.close();
        avgDocLength /= totalDocuments;
    }
    
    return true;
}

int main() {
    if (!loadIndex()) {
        cerr << "Error loading index\n";
        return 1;
    }
    
    cout << "Search engine ready. Type 'quit' to exit.\n";
    cout << "Prefix queries with 'AND:' for conjunctive, 'OR:' for disjunctive (default).\n\n";
    
    ifstream invFile("index/inverted_index.bin", ios::binary);
    if (!invFile.is_open()) {
        cerr << "Error opening inverted index\n";
        return 1;
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
        
        vector<pair<int, double>> results;
        if (conjunctive) {
            results = processConjunctiveQuery(invFile, queryTerms);
        } else {
            results = processDisjunctiveQuery(invFile, queryTerms);
        }
        
        cout << "\nTop 10 results:\n";
        for (int i = 0; i < min(10, (int)results.size()); i++) {
            cout << (i + 1) << ". DocID " << results[i].first 
                 << " (score: " << results[i].second << ")\n";
        }
        cout << "Total: " << results.size() << " documents\n\n";
    }
    
    invFile.close();
    return 0;
}