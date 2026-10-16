#include "rag/embedder/library.hpp"

#include "F16/f16.hpp"

#include <string>
#include <utility>
#include <vector>

using namespace Mln::Sdk;
using namespace Rag::Embedder::Arch;

namespace Rag::Embedder {
    std::vector<std::pair<std::string, Syn::Compiler::Police>> policies() {
        using namespace Syn::Compiler;

        std::vector<std::pair<std::string, Police>> policies;

        policies.emplace_back(
            "embedder", Police{
                            .schemes =
                                {
                                    Mln::Sdk::Codec::Scheme{
                                        .scope = Mln::Sdk::Codec::Scope::Constant,
                                        .lifetime = Mln::Sdk::Codec::Lifetime::Immutable,
                                        .update = Mln::Sdk::Codec::Update::None,
                                        .format = Mln::Sdk::Codec::Format(Quantizations::F16{}),
                                    },
                                },
                            .dtype = Mln::DType::Float16,
                            .backend =
                                Backend{
                                    .device = "cuda",
                                    .flags = {{"iree-cuda-target", "sm_86"}},
                                },
                        });

        return policies;
    }

    Syn::Compiler::Manifest manifest() {
        using namespace Syn::Compiler;

        Manifest built{
            .policies = policies(),
            .entrypoints =
                {
                    Entrypoint{
                        .name = "embedder.forward",
                        .specializations =
                            {
                                Specialization{
                                    .inputs =
                                        {
                                            Shape{{SEQUENCE}, Mln::DType::Float32},
                                            Shape{SEQUENCE},
                                        },
                                },
                            },
                    },
                },
        };

        built.formats.push_back(Mln::Sdk::Codec::Format(Quantizations::F16{}));
        return built;
    }
}
