/* Bridge for the external ggml reference library; never linked by Nathr. */
#include <ggml.h>

int nathr_reference_decode(enum ggml_type type, const void * packed, float * decoded, int64_t count) {
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    if (count <= 0 || count % traits->blck_size != 0 || traits->to_float == NULL) {
        return 0;
    }
    traits->to_float(packed, decoded, count);
    return 1;
}
