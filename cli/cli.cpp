#include "sam.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <sstream>
#include <iostream>

static bool load_image_from_file(const std::string & fname, sam_image_u8 & img) {
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }
    if (nc != 3) {
        fprintf(stderr, "%s: '%s' has %d channels (expected 3)\n", __func__, fname.c_str(), nc);
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

static void print_usage(int argc, char ** argv, const sam_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    fprintf(stderr, "                        input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    fprintf(stderr, "                        output file (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "\n");
}

static bool params_parse(int argc, char ** argv, sam_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_out = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

int main(int argc, char **argv) {
    struct sam_params params;
    if (!params_parse(argc, argv, params)) {
        return 1;
    }
    // params.model = "checkpoints/ggml-model-f16.bin";
    // params.fname_inp = "./img.jpg";
    // params.n_threads = 8;

    sam_image_u8 img0;
    std::vector<sam_image_u8> masks;
    if (!load_image_from_file(params.fname_inp, img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    std::shared_ptr<sam_state> state = sam_load_model(params);
    if (!sam_compute_embd_img(img0, params.n_threads, *state)) {

        fprintf(stderr, "%s: failed to compute encoded image\n", __func__);
        return 1;
    }
    float x = img0.nx / 2;
    float y = img0.ny / 2;
    sam_point pt { x, y};
    // fprintf(stderr, "x: %.2f, y: %.2f", x, y);
    masks = sam_compute_masks(img0, params.n_threads, pt, *state);
    int idx = 0;
    for (const auto& mask : masks) {
        std::stringstream n;
        std::stringstream p;
        n << "mask_" << idx << ".jpg";
        p << "seg_" << idx << ".jpg";
        std::cerr << "mask" << idx << std::endl;
        sam_image_u8 mask_rgb = { mask.nx, mask.ny, };
        mask_rgb.data.resize(3*mask.nx*mask.ny);
        for (int i = 0; i < mask.nx * mask.ny; ++i) {
            mask_rgb.data[3*i + 0] = mask.data[i] > 0 ? img0.data[3 * i + 0] : 0x0;
            mask_rgb.data[3*i + 1] = mask.data[i] > 0 ? img0.data[3 * i + 1] : 0x47;
            mask_rgb.data[3*i + 2] = mask.data[i] > 0 ? img0.data[3 * i + 2] : 0xab;
        }
        idx++;
        stbi_write_jpg(n.str().c_str(), mask.nx, mask.ny, 1 /*greyscale*/, reinterpret_cast<const void *>(mask.data.data()), 85);
        stbi_write_jpg(p.str().c_str(), mask.nx, mask.ny, 3 /*rgb*/, reinterpret_cast<const void *>(mask_rgb.data.data()), 85);
    }

    sam_deinit(*state);
}
