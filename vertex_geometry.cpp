#include "vertex_geometry.hpp"
#include <cassert>
#include <math.h>
#include "glm/geometric.hpp"
#include <algorithm>

/**
 *
 * @param out_square
 * @param center_x
 * @param center_y
 * @param side_length
 * @param iteration
 *
 * \todo Note that the triangles are done in clockwise ordering meaning they are
 *  back-facing so if you have culling turned on they will not show up. We should probably just make them counter-clockwise
 *  so that this doesn't occur.
 *
 *  note this is designed to work with store_square_indices inside of a drawElements call.
 */
void store_square_vertices(float *out_square, float center_x, float center_y, float side_length, int iteration) {

    store_rectangle_vertices(out_square, center_x, center_y, side_length, side_length, iteration);

//    float half_side_length = side_length / (float) 2;
//
//    // todo currently makes a square twice as big as side length
//    float generic_square_vertices[12] = {
//            center_x + half_side_length, center_y + half_side_length, 0.0f,  // top right
//            center_x + half_side_length, center_y - half_side_length , 0.0f,  // bottom right
//            center_x -half_side_length , center_y - half_side_length, 0.0f,  // bottom left
//            center_x -half_side_length, center_y + half_side_length, 0.0f   // top left
//    };
//
//    // copy into array
//    for (int i = 0; i < 12; i ++) {
//        out_square[i + (12 * iteration)] = generic_square_vertices[i];
//    }
}

void store_square_indices(unsigned int *out_indices, unsigned int iteration) {

    store_rectangle_indices(out_indices, iteration);

//    unsigned int indices[6] = {  // note that we start from 0!
//            0, 1, 3,   // first triangle
//            1, 2, 3    // second triangle
//    };
//
//    // assigns six values at once
//    for (int i = 0; i < 6; i ++ ) {
//        out_indices[i + (6 * iteration)] = indices[i] + (iteration * 4); // 4 is one larger than 1 makes disjoint
//    }
}

void store_rectangle_vertices(float *out_rectangle, float center_x, float center_y, float width, float height,
                              int iteration) {

    float half_width = width / (float) 2;
    float half_height = height / (float) 2;

    // todo currently makes a rectangle twice as big as side length
    float generic_rectangle_vertices[12] = {
            center_x + half_width, center_y + half_height, 0.0f,  // top right
            center_x + half_width, center_y - half_height, 0.0f,  // bottom right
            center_x - half_width, center_y - half_height, 0.0f,  // bottom left
            center_x - half_width, center_y + half_height, 0.0f   // top left
    };

    // copy into array
    for (int i = 0; i < 12; i++) {
        out_rectangle[i + (12 * iteration)] = generic_rectangle_vertices[i];
    }
}

void store_rectangle_indices(unsigned int *out_indices, unsigned int iteration) {

    unsigned int indices[6] = {  // note that we start from 0!
            0, 1, 3,   // first triangle
            1, 2, 3    // second triangle
    };

    // assigns six values at once
    for (int i = 0; i < 6; i++) {
        out_indices[i + (6 * iteration)] = indices[i] + (iteration * 4); // 4 is one larger than 1 makes disjoint
    }
}

int get_num_flattened_vertices_in_n_gon(int n) {
    // we have a central point, and then we repeat the first point again.
    int num_vertices = 1 + (n + 1);
    return num_vertices * 3;
}

/**
 *
 * \brief generate n equally spaced points on the unit circle
 *
 * Designed to work with glDrawArrays with a GL_TRIANGLE_FAN setting
 *
 * @param out_vertices
 *
 */
void store_n_gon_flattened_vertices(float *out_vertices, int n) {

    assert(n >= 3);

    out_vertices[0] = 0.0f;
    out_vertices[1] = 0.0f;
    out_vertices[2] = 0.0f;

    int origin_offset = 3, complete_offset;

    float angle_increment = (2 * M_PI) / (float) n;
    float curr_angle;

    // less than or equal to n, places the initial point at the start and end which is what we want.
    for (int i = 0; i <= n; i++) {
        complete_offset = origin_offset + 3 * i; // stride by 3, the size of a vertex
        curr_angle = i * angle_increment;
        out_vertices[complete_offset + 0] = cos(curr_angle);
        out_vertices[complete_offset + 1] = sin(curr_angle);
        out_vertices[complete_offset + 2] = 0.0f;
    }

}

/**
 * \brief Create the points of an arrow pointing from one point to another
 *
 *                                                g\
 *                                                | --\
 *                                                |    ---\
 *           |    a-------------------------------c        --\
 * stem      |    |                    -------/   |           --\
 * thickness |  start          -------/           |           -----end
 *           |    |    -------/                   |           --/
 *           |    b---/---------------------------d        --/
 *                                                |    ---/
 *                                                | --/
 *                                                f/
 *
 *                                                ------tip len-------
 *
 * returns [a, b, c, d, e, f, g] from this we want to produce indices
 *
 * {a, b, c}, {c, b, d}, {e, f, g}
 * {0, 1, 2}, {2, 1, 3}, {4, 5, 6}
 *
 * \author cuppajoeman (2024)
 */

void store_arrow_vertices(glm::vec2 start, glm::vec2 end, float stem_thickness, float tip_length,
                          float *out_flattened_vertices) {

    float half_thickness = stem_thickness / (float) 2;

    glm::vec2 start_to_end = end - start;
    glm::vec2 end_to_start = start - end;

    float stem_length = std::max(0.0f, glm::length(start_to_end) - tip_length);

    glm::vec2 start_to_end_normalized = glm::normalize(start_to_end);
    glm::vec2 end_to_start_normalized = glm::normalize(end_to_start);

    // note that these two are also normalized
    glm::vec2 start_to_a_dir = glm::vec2(-start_to_end_normalized.y, start_to_end_normalized.x);
    glm::vec2 start_to_b_dir = glm::vec2(start_to_end_normalized.y, -start_to_end_normalized.x);

    glm::vec2 a = start + start_to_a_dir * half_thickness;
    glm::vec2 b = start + start_to_b_dir * half_thickness;
    glm::vec2 c = a + start_to_end_normalized * stem_length;
    glm::vec2 d = b + start_to_end_normalized * stem_length;

    /*
     *
     *         (aln)--        |
     *               ----     |
     *                  ----  |
     *         (e2s)----------s-------------e
     *                  ----  |
     *               ----     |
     *         (bln)--        |
     *
     *         ------len 1-----
     */

    glm::vec2 a_line = end_to_start_normalized + start_to_a_dir;
    glm::vec2 b_line = end_to_start_normalized + start_to_b_dir;

    /*
     *       |o
     *       |    o    hyp = \sqrt{2} x
     *       x        o
     *       |            o
     *       |------x------- o
     */

    a_line = glm::normalize(a_line) * (float) sqrt(2) * tip_length;
    b_line = glm::normalize(b_line) * (float) sqrt(2) * tip_length;

    glm::vec2 g = end + a_line;
    glm::vec2 f = end + b_line;


    const int num_flattened_vertices_in_arrow = 3 * 7;

    float arrow[num_flattened_vertices_in_arrow] = {
            a.x, a.y, 0.0f,
            b.x, b.y, 0.0f,
            c.x, c.y, 0.0f,
            d.x, d.y, 0.0f,
            end.x, end.y, 0.0f,
            f.x, f.y, 0.0f,
            g.x, g.y, 0.0f
    };

    for (int i = 0; i < num_flattened_vertices_in_arrow; i++) {
        out_flattened_vertices[i] = arrow[i];
    }

}

void store_arrow_indices(unsigned int *out_indices) {
    const int num_vertices = 3 * 3; // only three triangles

    unsigned int indices[num_vertices] = {
            0, 1, 2,
            2, 1, 3,
            4, 5, 6
    };

    for (int i = 0; i < num_vertices; i++) {
        out_indices[i] = indices[i];
    }

}

// TODO eventually just use matrices in opengl for this.
void scale_vertices(float *vertices_to_be_scaled, int num_vertices, float scale_factor) {
    for (int i = 0; i < num_vertices; i++) {
        vertices_to_be_scaled[i] = vertices_to_be_scaled[i] * scale_factor;
    }
}

void
translate_vertices(float *flattened_vertices_to_be_scaled, int num_vertices, float x_translate, float y_translate) {
    float translations[3] = {x_translate, y_translate, 0.0f};
    for (int i = 0; i < num_vertices; i++) {
        flattened_vertices_to_be_scaled[i] = flattened_vertices_to_be_scaled[i] + translations[i % 3];
    }
}
