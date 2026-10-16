#include <mln/sdk/native.hpp>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
    constexpr int64_t MAX_RESULTS = 5;
    constexpr int64_t MAX_PASSAGES = 6;
    constexpr std::size_t PASSAGE_CHARS = 900;
    constexpr long HTTP_TIMEOUT_SECONDS = 15;
    constexpr curl_off_t MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024;

    constexpr const char *HOSTS[] = {
        "https://pt.wikipedia.org",
        "https://en.wikipedia.org",
    };

    std::once_flag curl_once;

    struct Document {
            std::string title;
            std::string url;
            std::string text;
    };

    struct Passage {
            std::string text;
            const Document *document;
            double score = 0.0;
    };

    std::size_t append_body(char *data, std::size_t size, std::size_t count, void *user) {
        auto *out = static_cast<std::string *>(user);
        const std::size_t total = size * count;
        if (out->size() + total > static_cast<std::size_t>(MAX_DOWNLOAD_BYTES))
            return 0;
        out->append(data, total);
        return total;
    }

    std::string http_get(const std::string &url) {
        std::call_once(curl_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
        CURL *handle = curl_easy_init();
        if (!handle)
            return {};

        std::string body;
        curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, append_body);
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, &body);
        curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 5L);
        curl_easy_setopt(handle, CURLOPT_TIMEOUT, HTTP_TIMEOUT_SECONDS);
        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 6L);
        curl_easy_setopt(handle, CURLOPT_ACCEPT_ENCODING, "");
        curl_easy_setopt(handle, CURLOPT_MAXFILESIZE_LARGE, MAX_DOWNLOAD_BYTES);

        curl_easy_setopt(handle, CURLOPT_USERAGENT, "MyelinRag/1.0 (Myelin RAG example)");
        const CURLcode status = curl_easy_perform(handle);
        curl_easy_cleanup(handle);
        return status == CURLE_OK ? body : std::string{};
    }

    std::string url_encode(std::string_view value) {
        std::string encoded;
        encoded.reserve(value.size() * 3);
        for (unsigned char character : value) {
            if (std::isalnum(character) || character == '-' || character == '_' ||
                character == '.' || character == '~') {
                encoded.push_back(static_cast<char>(character));
            } else {
                static constexpr char digits[] = "0123456789ABCDEF";
                encoded.push_back('%');
                encoded.push_back(digits[character >> 4]);
                encoded.push_back(digits[character & 0x0f]);
            }
        }
        return encoded;
    }

    std::vector<Document> search_host(const std::string &host, const std::string &query) {
        const std::string listing =
            http_get(host +
                     "/w/api.php?action=query&list=search&format=json"
                     "&srlimit=" +
                     std::to_string(MAX_RESULTS) + "&srsearch=" + url_encode(query));
        if (listing.empty())
            return {};

        std::vector<Document> documents;
        std::string ids;
        try {
            const auto parsed = nlohmann::json::parse(listing);
            for (const auto &entry : parsed.at("query").at("search")) {
                const auto pageid = entry.at("pageid").get<int64_t>();
                Document document;
                document.title = entry.value("title", "");
                document.url = host + "/?curid=" + std::to_string(pageid);
                documents.push_back(std::move(document));
                if (!ids.empty())
                    ids += "|";
                ids += std::to_string(pageid);
            }
        } catch (const nlohmann::json::exception &) {
            return {};
        }
        if (documents.empty())
            return {};

        const std::string extracts = http_get(host +
                                              "/w/api.php?action=query&prop=extracts&explaintext=1"
                                              "&format=json&pageids=" +
                                              ids);
        if (extracts.empty())
            return {};

        try {
            const auto parsed = nlohmann::json::parse(extracts);
            const auto &pages = parsed.at("query").at("pages");
            for (Document &document : documents) {
                for (const auto &page : pages) {
                    if (page.value("title", "") != document.title)
                        continue;
                    document.text = page.value("extract", "");
                    break;
                }
            }
        } catch (const nlohmann::json::exception &) {
            return {};
        }

        documents.erase(
            std::remove_if(documents.begin(), documents.end(),
                           [](const Document &document) { return document.text.size() < 200; }),
            documents.end());
        return documents;
    }

    std::vector<Document> search_documents(const std::string &query) {
        for (const char *host : HOSTS) {
            std::vector<Document> documents = search_host(host, query);
            if (!documents.empty())
                return documents;
        }
        return {};
    }

    std::vector<std::string> tokenize_words(std::string_view text) {
        std::vector<std::string> words;
        std::string current;
        for (unsigned char character : text) {
            if (std::isalnum(character)) {
                current.push_back(static_cast<char>(std::tolower(character)));
            } else if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
        }
        if (!current.empty())
            words.push_back(std::move(current));
        return words;
    }

    std::vector<Passage> split_passages(const std::vector<Document> &documents) {
        std::vector<Passage> passages;
        for (const Document &document : documents) {
            for (std::size_t start = 0; start < document.text.size(); start += PASSAGE_CHARS) {
                std::size_t length = std::min(PASSAGE_CHARS, document.text.size() - start);
                if (start + length < document.text.size()) {
                    const std::size_t space = document.text.rfind(' ', start + length);
                    if (space != std::string::npos && space > start)
                        length = space - start;
                }
                passages.push_back(Passage{document.text.substr(start, length), &document, 0.0});
            }
        }
        return passages;
    }

    void score_passages(std::vector<Passage> &passages,
                        const std::vector<std::string> &query_words) {
        std::unordered_map<std::string, int64_t> document_frequency;
        for (const Passage &passage : passages) {
            std::unordered_set<std::string> unique;
            for (const std::string &word : tokenize_words(passage.text))
                unique.insert(word);
            for (const std::string &word : unique)
                ++document_frequency[word];
        }

        const double total = static_cast<double>(passages.size());
        for (Passage &passage : passages) {
            std::unordered_map<std::string, int64_t> counts;
            for (const std::string &word : tokenize_words(passage.text))
                ++counts[word];
            const double length =
                std::max<double>(1.0, static_cast<double>(passage.text.size()) / PASSAGE_CHARS);

            double score = 0.0;
            for (const std::string &word : query_words) {
                const auto found = counts.find(word);
                if (found == counts.end())
                    continue;
                const double frequency = static_cast<double>(document_frequency[word]);
                const double idf = std::log((total + 1.0) / (frequency + 0.5));
                score += idf * (1.0 + std::log(static_cast<double>(found->second)));
            }
            passage.score = score / length;
        }
    }
}

PUBLISH(rag, search, std::vector<uint8_t>, std::span<const uint8_t> query) {
    const std::string question(reinterpret_cast<const char *>(query.data()), query.size());
    if (question.empty())
        return {};

    const std::vector<Document> documents = search_documents(question);
    if (documents.empty())
        return {};

    std::vector<Passage> passages = split_passages(documents);
    if (passages.empty())
        return {};

    score_passages(passages, tokenize_words(question));
    const std::size_t keep = std::min<std::size_t>(MAX_PASSAGES, passages.size());
    std::partial_sort(
        passages.begin(), passages.begin() + keep, passages.end(),
        [](const Passage &left, const Passage &right) { return left.score > right.score; });

    std::string context;
    for (std::size_t i = 0; i < keep; ++i) {
        if (passages[i].score <= 0.0)
            break;
        context += "[" + std::to_string(i + 1) + "] " + passages[i].document->title + " (" +
                   passages[i].document->url + ")\n" + passages[i].text + "\n\n";
    }
    return std::vector<uint8_t>(context.begin(), context.end());
}
