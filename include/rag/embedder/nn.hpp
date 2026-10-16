#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

#include <cstdint>
#include <string>

namespace Rag::Embedder {
    namespace Arch {
        inline constexpr int64_t HIDDEN_SIZE = 384;
        inline constexpr int64_t NUM_LAYERS = 12;
        inline constexpr int64_t NUM_HEADS = 12;
        inline constexpr int64_t HEAD_DIM = 32;
        inline constexpr int64_t INTERMEDIATE_SIZE = 1536;
        inline constexpr int64_t VOCAB_SIZE = 250037;
        inline constexpr int64_t MAX_POSITIONS = 512;
        inline constexpr double LAYER_NORM_EPS = 1e-12;

        inline constexpr int64_t SEQUENCE = 256;

        inline constexpr int64_t BOS_TOKEN_ID = 0;
        inline constexpr int64_t PAD_TOKEN_ID = 1;
        inline constexpr int64_t EOS_TOKEN_ID = 2;
        inline constexpr int64_t UNK_TOKEN_ID = 3;
    }

    class Dense : public Mln::Sdk::Module {
        public:
            Dense(std::string prefix, int64_t inputs, int64_t outputs);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            std::string prefix_;
            int64_t inputs_;
            int64_t outputs_;
    };

    class LayerNorm : public Mln::Sdk::Module {
        public:
            explicit LayerNorm(std::string prefix);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            std::string prefix_;
    };

    class Attention : public Mln::Sdk::Module {
        public:
            explicit Attention(int64_t layer);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &hidden, const Mln::Sdk::Tensor &bias);
    };

    class Layer : public Mln::Sdk::Module {
        public:
            explicit Layer(int64_t layer);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &hidden, const Mln::Sdk::Tensor &bias);
    };

    class Encoder final : public Mln::Sdk::Module {
        public:
            Encoder();

            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &tokens, const Mln::Sdk::Tensor &mask);

            std::vector<Mln::Sdk::Tensor> graph(std::span<const Mln::Sdk::Tensor> inputs) override;
            std::vector<std::vector<Mln::Sdk::Tensor>> specializations() const override;
    };
}
