#pragma once

#include "rag/embedder/nn.hpp"

#include <mln/sdk/orchestration.hpp>

namespace Rag::Embedder {
    class Orchestration : public Mln::Sdk::Orchestration {
        public:
            Orchestration() {
                name = "e5";
                modules.insert<Encoder>("embedder");
            }

            Mln::Buffer forward(Mln::Runtime::Session &session, Mln::Buffer input) override;
    };
}
