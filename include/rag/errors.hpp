#pragma once

#include <mln/error.hpp>

namespace Rag::Errors {
    namespace Embedder::Graph::Input::Count {
        inline constexpr Mln::Error Invalid{4200, "RAG embedder expected 2 inputs, received {}"};
    }

    namespace Hook::Embedding::Output::Count {
        inline constexpr Mln::Error Invalid{4201, "RAG hook expected 1 embedding, received {}"};
    }

    namespace Hook::Search::Result {
        namespace Count {
            inline constexpr Mln::Error Invalid{4202,
                                                "RAG search returned {} results for a limit of {}"};
        }
        namespace Storage {
            inline constexpr Mln::Error Invalid{
                4203, "RAG search returned {} bytes; expected {} for {} results"};
        }
    }

    namespace Embedder::Runtime::Output::Count {
        inline constexpr Mln::Error Invalid{4204,
                                            "RAG embedder runtime expected 1 output, received {}"};
    }

    namespace Index {
        namespace Handle {
            inline constexpr Mln::Error Null{4205, "RAG received a null index handle"};
            inline constexpr Mln::Error Unknown{4206, "RAG did not find index handle {}"};
        }
        namespace Asset {
            inline constexpr Mln::Error Empty{4207, "RAG received an empty index asset"};
            namespace Load {
                inline constexpr Mln::Error Failed{4208, "RAG could not load the index: {}"};
            }
        }
        namespace Create {
            inline constexpr Mln::Error Failed{4209, "RAG could not create the index: {}"};
        }
        namespace Dimension {
            inline constexpr Mln::Error Zero{4210, "RAG received a zero index dimension"};
        }
        namespace Add {
            namespace Payload {
                inline constexpr Mln::Error Invalid{
                    4211, "RAG received {} bytes that do not encode vectors and identifiers"};
            }
            inline constexpr Mln::Error Failed{4212, "RAG could not insert the vectors: {}"};
        }
        namespace Remove {
            inline constexpr Mln::Error Failed{4213, "RAG could not remove the vector: {}"};
        }
        namespace Search {
            namespace Limit {
                inline constexpr Mln::Error Zero{4214, "RAG received a zero search limit"};
            }
            namespace Query {
                inline constexpr Mln::Error Invalid{
                    4215, "RAG received a {}-byte query that does not encode float32 values"};
            }
            inline constexpr Mln::Error Failed{4216, "RAG could not execute the search: {}"};
        }
        namespace Serialization {
            inline constexpr Mln::Error Failed{4217, "RAG could not serialize the index: {}"};
            namespace Size {
                inline constexpr Mln::Error Invalid{
                    4218, "RAG serialized {} bytes, but {} were expected"};
            }
        }
    }

    namespace Embedder::Tokenizer {
        namespace Json {
            inline constexpr Mln::Error Invalid{4219, "RAG tokenizer contains invalid JSON"};
        }
        namespace Model {
            inline constexpr Mln::Error Unsupported{
                4220, "RAG tokenizer uses model '{}'; expected Unigram"};
        }
        namespace Vocabulary {
            inline constexpr Mln::Error Empty{4221,
                                              "RAG Unigram tokenizer has an empty vocabulary"};
        }
    }
}
