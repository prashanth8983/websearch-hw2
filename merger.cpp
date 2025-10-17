#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>

using namespace std;

const int BLOCK_SIZE = 128;

struct TermEntry {
    string term;
    int docID;
    int freq;
    int fileIndex;
    
    bool operator>(const TermEntry& other) const {
        if (term != other.term) return term > other.term;
        return docID > other.docID;
    }
};

void varbyte_encode(int num, vector<unsigned char>& output) {
    while (num >= 128) {
        output.push_back((num & 0x7F) | 0x80);
        num >>= 7;
    }
    output.push_back(num & 0x7F);
}

void write_block(ofstream& invFile, const vector<int>& docIDs, const vector<int>& freqs,vector<int>& lastDocIDs, vector<int>& docIDSizes, vector<int>& freqSizes) {
    
    // Delta encode docIDs
    vector<int> deltas;
    deltas.push_back(docIDs[0]);
    for (size_t i = 1; i < docIDs.size(); i++) {
        deltas.push_back(docIDs[i] - docIDs[i-1]);
    }
    
    // Varbyte encode
    vector<unsigned char> encodedDocIDs, encodedFreqs;
    for (int delta : deltas) {
        varbyte_encode(delta, encodedDocIDs);
    }
    for (int freq : freqs) {
        varbyte_encode(freq, encodedFreqs);
    }
    
    // Write blocks
    int docIDSize = encodedDocIDs.size();
    invFile.write(reinterpret_cast<const char*>(&docIDSize), sizeof(int));
    invFile.write(reinterpret_cast<const char*>(encodedDocIDs.data()), docIDSize);
    
    int freqSize = encodedFreqs.size();
    invFile.write(reinterpret_cast<const char*>(&freqSize), sizeof(int));
    invFile.write(reinterpret_cast<const char*>(encodedFreqs.data()), freqSize);
    
    // Store metadata
    lastDocIDs.push_back(docIDs.back());
    docIDSizes.push_back(docIDSize);
    freqSizes.push_back(freqSize);
}

// Read next items from binary run file
bool read_next(ifstream& file, string& term, int& docID, int& freq) {
    // Read term length
    int termLen;
    if (!file.read(reinterpret_cast<char*>(&termLen), sizeof(int))) {
        return false;
    }
    
    // Read term
    term.resize(termLen);
    file.read(&term[0], termLen);
    
    // Read docID and freq
    file.read(reinterpret_cast<char*>(&docID), sizeof(int));
    file.read(reinterpret_cast<char*>(&freq), sizeof(int));
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <num_runs>\n";
        return 1;
    }
    
    int numRuns = stoi(argv[1]);
    
    // Open all run files
    vector<ifstream> runFiles(numRuns);
    for (int i = 0; i < numRuns; i++) {
        string filename = "partial/run_" + to_string(i) + ".bin";
        runFiles[i].open(filename, ios::binary);
        if (!runFiles[i].is_open()) {
            cerr << "Error: Cannot open " << filename << "\n";
            return 1;
        }
    }
    
    // Open output files
    ofstream invFile("index/inverted_index.bin", ios::binary);
    ofstream lexFile("index/lexicon.txt");
    
    if (!invFile.is_open() || !lexFile.is_open()) {
        cerr << "Error: Cannot create output files\n";
        return 1;
    }
    
    // Min heap - Priority queue for k-way merge
    priority_queue<TermEntry, vector<TermEntry>, greater<TermEntry>> pq;
    
    // Initialize
    vector<string> currentTerms(numRuns);
    vector<int> currentDocIDs(numRuns);
    vector<int> currentFreqs(numRuns);
    
    for (int i = 0; i < numRuns; i++) {
        if (read_next(runFiles[i], currentTerms[i], currentDocIDs[i], currentFreqs[i])) {
            pq.push({currentTerms[i], currentDocIDs[i], currentFreqs[i], i});
        }
    }
    
    // Metadata arrays
    vector<int> allLastDocIDs, allDocIDSizes, allFreqSizes;
    
    // Statistics
    int termsProcessed = 0;
    long long totalPostings = 0;
    unordered_map<string, int> termDF;
    
    // Current term
    string currentTerm = "";
    vector<int> termDocIDs, termFreqs;
    long long termStartOffset = 0;
    int termStartBlock = 0;
    
    while (!pq.empty()) {
        TermEntry entry = pq.top();
        pq.pop();
        
        // New term - finish previous
        if (!currentTerm.empty() && entry.term != currentTerm) {
            if (!termDocIDs.empty()) {
                write_block(invFile, termDocIDs, termFreqs, 
                          allLastDocIDs, allDocIDSizes, allFreqSizes);
            }
            
            // Write lexicon entry
            lexFile << currentTerm << "\t" 
                    << termStartOffset << "\t"
                    << termStartBlock << "\t"
                    << totalPostings << "\t"
                    << termDF[currentTerm] << "\n";
            
            termDocIDs.clear();
            termFreqs.clear();
            termStartOffset = invFile.tellp();
            termStartBlock = allLastDocIDs.size();
            termsProcessed++;
            
            if (termsProcessed % 50000 == 0) {
                cerr << "Merged " << termsProcessed << " terms\r" << flush;
            }
        }
        
        // Process current posting
        if (entry.term != currentTerm) {
            currentTerm = entry.term;
            termDF[currentTerm] = 0;
            totalPostings = 0;
        }
        
        termDocIDs.push_back(entry.docID);
        termFreqs.push_back(entry.freq);
        termDF[currentTerm]++;
        totalPostings++;
        
        // Write block when full
        if ((int)termDocIDs.size() == BLOCK_SIZE) {
            write_block(invFile, termDocIDs, termFreqs,
                      allLastDocIDs, allDocIDSizes, allFreqSizes);
            termDocIDs.clear();
            termFreqs.clear();
        }
        
        // Read next from same file
        int fileIdx = entry.fileIndex;
        if (read_next(runFiles[fileIdx], currentTerms[fileIdx], 
                           currentDocIDs[fileIdx], currentFreqs[fileIdx])) {
            pq.push({currentTerms[fileIdx], currentDocIDs[fileIdx], 
                    currentFreqs[fileIdx], fileIdx});
        }
    }
    
    // Process last term
    if (!currentTerm.empty()) {
        if (!termDocIDs.empty()) {
            write_block(invFile, termDocIDs, termFreqs,
                      allLastDocIDs, allDocIDSizes, allFreqSizes);
        }
        
        lexFile << currentTerm << "\t" 
                << termStartOffset << "\t"
                << termStartBlock << "\t"
                << totalPostings << "\t"
                << termDF[currentTerm] << "\n";
        termsProcessed++;
    }
    
    // Close files
    for (auto& f : runFiles) f.close();
    invFile.close();
    lexFile.close();
    
    // Write metadata
    ofstream metaFile("index/metadata.bin", ios::binary);
    
    int numBlocks = allLastDocIDs.size();
    metaFile.write(reinterpret_cast<const char*>(&numBlocks), sizeof(int));
    metaFile.write(reinterpret_cast<const char*>(allLastDocIDs.data()), numBlocks * sizeof(int));
    metaFile.write(reinterpret_cast<const char*>(allDocIDSizes.data()), numBlocks * sizeof(int));
    metaFile.write(reinterpret_cast<const char*>(allFreqSizes.data()),  numBlocks * sizeof(int));
    metaFile.close();
    
    // Write stats
    ofstream statsFile("index/collection_stats.txt");
    statsFile << "total_terms\t" << termsProcessed << "\n";
    statsFile << "total_blocks\t" << numBlocks << "\n";
    statsFile.close();
    
    cerr << "\nMerge complete: " << termsProcessed << " terms, " << numBlocks << " blocks\n";
    
    return 0;
}