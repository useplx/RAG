#pragma once

#include <mln/buffer.hpp>
#include <mln/runtime/runtime.hpp>

namespace Rag::Hook {
    Mln::Buffer augment(Mln::Runtime::Session &session, Mln::Buffer input);
}
