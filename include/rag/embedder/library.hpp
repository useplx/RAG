#pragma once

#include "rag/embedder/orchestration.hpp"

#include <syn/manifest.hpp>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace Rag::Embedder {
    std::vector<std::pair<std::string, Syn::Compiler::Police>> policies();

    Syn::Compiler::Manifest manifest();
}
