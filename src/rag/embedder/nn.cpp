#include "rag/embedder/nn.hpp"
#include "rag/errors.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/nn/gelu.hpp>
#include <mln/sdk/operations.hpp>

#include <utility>
#include <vector>

using namespace Mln::Sdk;
using namespace Rag::Embedder::Arch;

namespace Rag::Embedder {
    namespace {
        Tensor scalar(Operations &operations, double value, std::vector<int64_t> shape) {
            Tensor constant = operations.insert<Functional::Const>(Weight{"", {}, value}, true);
            return operations.insert<Functional::Broadcast>(std::move(constant), std::move(shape),
                                                            std::vector<int64_t>{});
        }

        Tensor spread(Operations &operations, Tensor rows, int64_t count, int64_t columns) {
            return operations.insert<Functional::Broadcast>(
                std::move(rows), std::vector<int64_t>{count, columns}, std::vector<int64_t>{0});
        }

        Tensor tile(Operations &operations, Tensor vector, int64_t rows, int64_t columns) {
            return operations.insert<Functional::Broadcast>(
                std::move(vector), std::vector<int64_t>{rows, columns}, std::vector<int64_t>{1});
        }
    }

    Dense::Dense(std::string prefix, int64_t inputs, int64_t outputs)
        : prefix_(std::move(prefix)), inputs_(inputs), outputs_(outputs) {}

    Tensor Dense::forward(const Tensor &x) {
        Tensor weight = operations.insert<Functional::Const>(
            Weight{prefix_ + ".weight", {outputs_, inputs_}}, true);
        Tensor bias =
            operations.insert<Functional::Const>(Weight{prefix_ + ".bias", {outputs_}}, true);

        Tensor transposed =
            operations.insert<Functional::Transpose>(std::move(weight), std::vector<int64_t>{1, 0});
        Tensor projected = operations.insert<Functional::MatMul>(x, std::move(transposed));

        const int64_t rows = x.shape.front();
        return operations.insert<Functional::Add>(
            std::move(projected), tile(operations, std::move(bias), rows, outputs_));
    }

    LayerNorm::LayerNorm(std::string prefix) : prefix_(std::move(prefix)) {}

    Tensor LayerNorm::forward(const Tensor &x) {
        const int64_t rows = x.shape.front();
        const int64_t columns = x.shape.back();

        Tensor weight =
            operations.insert<Functional::Const>(Weight{prefix_ + ".weight", {columns}}, true);
        Tensor bias =
            operations.insert<Functional::Const>(Weight{prefix_ + ".bias", {columns}}, true);

        Tensor inverse = scalar(operations, 1.0 / static_cast<double>(columns), {rows, columns});
        Tensor mean = operations.insert<Functional::ReduceSum>(
            operations.insert<Functional::Multiply>(x, inverse), std::vector<int64_t>{1});
        Tensor centered = operations.insert<Functional::Subtract>(
            x, spread(operations, std::move(mean), rows, columns));

        Tensor variance = operations.insert<Functional::ReduceSum>(
            operations.insert<Functional::Multiply>(
                operations.insert<Functional::Multiply>(centered, centered), std::move(inverse)),
            std::vector<int64_t>{1});
        Tensor deviation = operations.insert<Functional::Rsqrt>(operations.insert<Functional::Add>(
            std::move(variance), scalar(operations, LAYER_NORM_EPS, {rows})));

        Tensor normalized = operations.insert<Functional::Multiply>(
            std::move(centered), spread(operations, std::move(deviation), rows, columns));

        return operations.insert<Functional::Add>(
            operations.insert<Functional::Multiply>(
                std::move(normalized), tile(operations, std::move(weight), rows, columns)),
            tile(operations, std::move(bias), rows, columns));
    }

    Attention::Attention(int64_t layer) {
        const std::string base = "encoder.layer." + std::to_string(layer) + ".attention";
        modules.insert<Dense>("query", base + ".self.query", HIDDEN_SIZE, HIDDEN_SIZE);
        modules.insert<Dense>("key", base + ".self.key", HIDDEN_SIZE, HIDDEN_SIZE);
        modules.insert<Dense>("value", base + ".self.value", HIDDEN_SIZE, HIDDEN_SIZE);
        modules.insert<Dense>("output", base + ".output.dense", HIDDEN_SIZE, HIDDEN_SIZE);
        modules.insert<LayerNorm>("norm", base + ".output.LayerNorm");
    }

    Tensor Attention::forward(const Tensor &hidden, const Tensor &bias) {
        auto heads = [&](Tensor projected) {
            Tensor split = operations.insert<Functional::Reshape>(
                std::move(projected), std::vector<int64_t>{SEQUENCE, NUM_HEADS, HEAD_DIM});
            return operations.insert<Functional::Transpose>(std::move(split),
                                                            std::vector<int64_t>{1, 0, 2});
        };

        Tensor q = heads(modules.forward<Dense>("query", hidden));
        Tensor k = heads(modules.forward<Dense>("key", hidden));
        Tensor v = heads(modules.forward<Dense>("value", hidden));

        Tensor scores = operations.insert<Functional::MatMul>(
            std::move(q),
            operations.insert<Functional::Transpose>(std::move(k), std::vector<int64_t>{0, 2, 1}));

        Tensor scale = operations.insert<Functional::Broadcast>(
            operations.insert<Functional::Const>(Weight{"", {}, 1.0 / 5.656854249492381}, true),
            std::vector<int64_t>{NUM_HEADS, SEQUENCE, SEQUENCE}, std::vector<int64_t>{});
        scores = operations.insert<Functional::Multiply>(std::move(scores), std::move(scale));

        scores = operations.insert<Functional::Add>(
            std::move(scores), operations.insert<Functional::Broadcast>(
                                   bias, std::vector<int64_t>{NUM_HEADS, SEQUENCE, SEQUENCE},
                                   std::vector<int64_t>{1, 2}));

        Tensor maximum = operations.insert<Functional::Broadcast>(
            operations.insert<Functional::ReduceMax>(scores, std::vector<int64_t>{2}),
            std::vector<int64_t>{NUM_HEADS, SEQUENCE, SEQUENCE}, std::vector<int64_t>{0, 1});
        Tensor weights = operations.insert<Functional::Exp>(
            operations.insert<Functional::Subtract>(std::move(scores), std::move(maximum)));
        Tensor total = operations.insert<Functional::Broadcast>(
            operations.insert<Functional::ReduceSum>(weights, std::vector<int64_t>{2}),
            std::vector<int64_t>{NUM_HEADS, SEQUENCE, SEQUENCE}, std::vector<int64_t>{0, 1});
        weights = operations.insert<Functional::Divide>(std::move(weights), std::move(total));

        Tensor context = operations.insert<Functional::MatMul>(std::move(weights), std::move(v));
        context = operations.insert<Functional::Transpose>(std::move(context),
                                                           std::vector<int64_t>{1, 0, 2});
        context = operations.insert<Functional::Reshape>(
            std::move(context), std::vector<int64_t>{SEQUENCE, HIDDEN_SIZE});

        Tensor projected = modules.forward<Dense>("output", context);
        return modules.forward<LayerNorm>(
            "norm", operations.insert<Functional::Add>(hidden, std::move(projected)));
    }

    Layer::Layer(int64_t layer) {
        const std::string base = "encoder.layer." + std::to_string(layer);
        modules.insert<Attention>("attention", layer);
        modules.insert<Dense>("intermediate", base + ".intermediate.dense", HIDDEN_SIZE,
                              INTERMEDIATE_SIZE);
        modules.insert<Mln::Sdk::Nn::GELU>("activation");
        modules.insert<Dense>("output", base + ".output.dense", INTERMEDIATE_SIZE, HIDDEN_SIZE);
        modules.insert<LayerNorm>("norm", base + ".output.LayerNorm");
    }

    Tensor Layer::forward(const Tensor &hidden, const Tensor &bias) {
        Tensor attended = modules.forward<Attention>("attention", hidden, bias);
        Tensor expanded = modules.forward<Dense>("intermediate", attended);
        expanded = modules.forward<Mln::Sdk::Nn::GELU>("activation", expanded);
        Tensor projected = modules.forward<Dense>("output", expanded);
        return modules.forward<LayerNorm>(
            "norm", operations.insert<Functional::Add>(std::move(attended), std::move(projected)));
    }

    Encoder::Encoder() {
        name = "embedder";
        modules.insert<LayerNorm>("embeddings", "embeddings.LayerNorm");
        for (int64_t layer = 0; layer < NUM_LAYERS; ++layer)
            modules.insert<Layer>("layers." + std::to_string(layer), layer);
    }

    std::vector<Mln::Sdk::Tensor> Encoder::graph(std::span<const Mln::Sdk::Tensor> inputs) {
        if (inputs.size() != 2)
            Mln::raise(Errors::Embedder::Graph::Input::Count::Invalid, inputs.size());
        return {forward(inputs[0], inputs[1])};
    }

    std::vector<std::vector<Tensor>> Encoder::specializations() const {
        return {{Tensor({SEQUENCE}, Mln::DType::Float32), Tensor({SEQUENCE}, Mln::DType::Float16)}};
    }

    Tensor Encoder::forward(const Tensor &tokens, const Tensor &mask) {
        Tensor ids = operations.insert<Functional::Convert>(tokens, Mln::DType::Int32);
        Tensor words = operations.insert<Functional::Gather>(
            operations.insert<Functional::Const>(
                Weight{"embeddings.word_embeddings.weight", {VOCAB_SIZE, HIDDEN_SIZE}}, true),
            std::move(ids), 0);

        Tensor positions = operations.insert<Functional::Slice>(
            operations.insert<Functional::Const>(
                Weight{"embeddings.position_embeddings.weight", {MAX_POSITIONS, HIDDEN_SIZE}},
                true),
            std::vector<int64_t>{0, 0}, std::vector<int64_t>{SEQUENCE, HIDDEN_SIZE},
            std::vector<int64_t>{1, 1});

        Tensor types = operations.insert<Functional::Slice>(
            operations.insert<Functional::Const>(
                Weight{"embeddings.token_type_embeddings.weight", {2, HIDDEN_SIZE}}, true),
            std::vector<int64_t>{0, 0}, std::vector<int64_t>{1, HIDDEN_SIZE},
            std::vector<int64_t>{1, 1});
        types = operations.insert<Functional::Broadcast>(
            operations.insert<Functional::Reshape>(std::move(types),
                                                   std::vector<int64_t>{HIDDEN_SIZE}),
            std::vector<int64_t>{SEQUENCE, HIDDEN_SIZE}, std::vector<int64_t>{1});

        Tensor hidden = operations.insert<Functional::Add>(
            operations.insert<Functional::Add>(std::move(words), std::move(positions)),
            std::move(types));
        hidden = modules.forward<LayerNorm>("embeddings", hidden);

        Tensor bias = operations.insert<Functional::Multiply>(
            operations.insert<Functional::Subtract>(mask, scalar(operations, 1.0, {SEQUENCE})),
            scalar(operations, 1.0e4, {SEQUENCE}));
        bias = operations.insert<Functional::Broadcast>(
            std::move(bias), std::vector<int64_t>{SEQUENCE, SEQUENCE}, std::vector<int64_t>{1});

        for (int64_t layer = 0; layer < NUM_LAYERS; ++layer)
            hidden = modules.forward<Layer>("layers." + std::to_string(layer), hidden, bias);

        Tensor expanded = spread(operations, mask, SEQUENCE, HIDDEN_SIZE);
        Tensor summed = operations.insert<Functional::ReduceSum>(
            operations.insert<Functional::Multiply>(hidden, expanded), std::vector<int64_t>{0});
        Tensor valid = operations.insert<Functional::ReduceSum>(mask, std::vector<int64_t>{0});
        Tensor pooled = operations.insert<Functional::Divide>(
            std::move(summed),
            operations.insert<Functional::Broadcast>(
                std::move(valid), std::vector<int64_t>{HIDDEN_SIZE}, std::vector<int64_t>{}));

        Tensor wide =
            operations.insert<Functional::Convert>(std::move(pooled), Mln::DType::Float32);

        Tensor norm = operations.insert<Functional::Rsqrt>(operations.insert<Functional::Add>(
            operations.insert<Functional::ReduceSum>(
                operations.insert<Functional::Multiply>(wide, wide), std::vector<int64_t>{0}),
            operations.insert<Functional::Const>(Weight{"", {}, 1e-12}, false,
                                                 Mln::DType::Float32)));

        return operations.insert<Functional::Multiply>(
            std::move(wide),
            operations.insert<Functional::Broadcast>(
                std::move(norm), std::vector<int64_t>{HIDDEN_SIZE}, std::vector<int64_t>{}));
    }
}
