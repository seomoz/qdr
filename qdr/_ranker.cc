
#include <math.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>


typedef std::unordered_map<std::string, std::pair<uint64_t, uint64_t> >
    counts_t;


// count of word occurances in a doc
typedef std::unordered_map<std::string, uint64_t> word_counts_t;

// a tokenized document
typedef std::vector<std::string> doc_t;

word_counts_t count_words(const doc_t& document)
{
    /// Given a vector of tokens, return a map token -> count in vector
    word_counts_t ret(document.size() * 3);
    for (doc_t::const_iterator it = document.begin();
        it != document.end(); ++it)
    {
        word_counts_t::iterator got = ret.find(*it);
        if (got == ret.end())
        {
            // first time we've seen this word
            ret[*it] = 1;
        }
        else
            ret[*it] += 1;
    }
    return ret;
}

struct scores_t {
    double tfidf;
    double bm25;
};


class QDR
{
    public:
        QDR(counts_t& counts, uint64_t total_docs);
        ~QDR();

        // compute the similarity scores
        scores_t score(doc_t& document, doc_t& query);

        // get the IDF for a given word
        double get_idf(const std::string& word);

    private:
        counts_t counts;
        uint64_t total_docs;
        uint64_t nwords_vocab;
        uint64_t nwords;

        // compute the TF-IDF similarity
        double tfidf(const word_counts_t& doc_counts,
            const word_counts_t& query_counts);

        // disable some default constructors
        QDR();
        QDR& operator= (const QDR& other);
        QDR(const QDR& other);

};

QDR::QDR(counts_t& counts_in, uint64_t total_docs)
{
    this->total_docs = total_docs;

    // copy the counts data to private member and add up the total
    // number of words and words in vocab
    nwords_vocab = 0;
    nwords = 0;
    counts.clear();

    for (counts_t::iterator it = counts_in.begin(); it != counts_in.end();
        ++it)
    {
        counts[it->first] = it->second;
        ++nwords_vocab;
        nwords += it->second.first;
    }
}

QDR::~QDR() {}

double QDR::get_idf(const std::string& word)
{
    /// Get log(total docs / doc count for word) == IDF
    double doc_freq;
    counts_t::iterator got = counts.find(word);
    if (got != counts.end())
    {
        // this word is in the corpus
        doc_freq = (double) got->second.second;
    }
    else
        doc_freq = 1.0;
    return log((double) total_docs / doc_freq);
}

scores_t QDR::score(doc_t& document, doc_t& query)
{
    word_counts_t query_counts = count_words(query);
    word_counts_t doc_counts = count_words(document);
    scores_t scores;
    scores.tfidf = tfidf(doc_counts, query_counts);
    scores.bm25 = 12.1;
    return scores;
}

double QDR::tfidf(
    const word_counts_t& doc_counts, const word_counts_t& query_counts)
{
    /// Compute TF-IDF similarity between the document and query
    // Computes the cosine similarity score of the query and document
    // using the "best fully weighted system tfc * nfx" (Table 2, first line)
    // from Salton and Buckley, "Term-weighting approaches in automatic text
    // retrieval", Information Processing & Mangement, vol 24, 1988"""

    // strategy: make document and query vectors then take inner product
    // only need to make the document vector for each query word
    // since other entries are zeroed out in inner product
    std::vector<double> doc_vector(query_counts.size());
    std::vector<double> query_vector(query_counts.size());

    // need the maximum count of query word occurances
    double max_query = 0.0;
    for (word_counts_t::const_iterator it_query = query_counts.begin();
        it_query != query_counts.end(); ++it_query)
    {
        if (it_query->second > max_query)
            max_query = it_query->second;
    }
//    std::cout << "max_query: " << max_query << "\n";

    // now make the vectors
    for (word_counts_t::const_iterator it_query = query_counts.begin();
        it_query != query_counts.end(); ++it_query)
    {
        // compute the idf
        double idf = get_idf(it_query->first);

        // query vector
        query_vector.push_back(
            (0.5 + 0.5 * it_query->second / max_query) * idf);

        // doc vector - if word is in doc then compute it, otherwise it's 0
        word_counts_t::const_iterator got = doc_counts.find(it_query->first);
        if (got != doc_counts.end())
            doc_vector.push_back(idf * got->second);
        else
            doc_vector.push_back(0.0);
    }

    // need to normalize the document vector with the euclidian length
    // the doc_vector variable only includes elements in the doc vector
    // corresponding to query words but we need the length of the full
    // vector here.  therefore need to iterate through doc_counts
    double doc_vector_len = 0.0;
    for (word_counts_t::const_iterator it_doc = doc_counts.begin();
        it_doc != doc_counts.end(); ++it_doc)
    {
        double p = ((double) it_doc->second) * get_idf(it_doc->first);
        doc_vector_len += p * p;
    }
    doc_vector_len = sqrt(doc_vector_len);

    // finally compute the score
    double score = 0.0;
    for (std::size_t k = 0; k < doc_vector.size(); ++k)
        score += query_vector[k] * doc_vector[k];
    return score / doc_vector_len;
}

