#include "rag/runtime_hook.hpp"

#include <mln/sdk/native.hpp>

namespace Rag::Hook {
    class Orchestration final : public Mln::Sdk::Orchestration {
        public:
            Mln::Buffer forward(Mln::Runtime::Session &session, Mln::Buffer input) override {
                return augment(session, std::move(input));
            }
    };
}

ENTRYPOINT(Rag::Hook::Orchestration);
