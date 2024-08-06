#include "vertex_geometry.hpp"
#include <cassert>
#include <math.h>
#include "glm/geometric.hpp"
#include <algorithm>

/**
 *
 * @param center_x
 * @param center_y
 * @param side_length
 *
 * \todo Note that the triangles are done in clockwise ordering meaning they are back-facing so if you have culling
 * turned on they will not show up. We should probably just make them counter-clockwise so that this doesn't occur.
 *
 *  note this is designed to work with generate_square_indices inside of a drawElements call.
 */
std::vector<glm::vec3> generate_square_vertices(float center_x, float center_y, float side_length) {
    return generate_rectangle_vertices(center_x, center_y, side_length, side_length);
}

std::vector<unsigned int> generate_square_indices() { return generate_rectangle_indices(); }

std::vector<glm::vec3> generate_rectangle_vertices(float center_x, float center_y, float width, float height) {

    float half_width = width / (float)2;
    float half_height = height / (float)2;

    // todo currently makes a rectangle twice as big as side length
    return {
        {center_x + half_width, center_y + half_height, 0.0f}, // top right
        {center_x + half_width, center_y - half_height, 0.0f}, // bottom right
        {center_x - half_width, center_y - half_height, 0.0f}, // bottom left
        {center_x - half_width, center_y + half_height, 0.0f}  // top left
    };
}

std::vector<unsigned int> generate_rectangle_indices() {
    return {
        // note that we start from 0!
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
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
 */
std::vector<glm::vec3> generate_n_gon_flattened_vertices(int n) {
    assert(n >= 3);
    float angle_increment = (2 * M_PI) / (float)n;
    float curr_angle;

    std::vector<glm::vec3> n_gon_points;
    // less than or equal to n, places the initial point at the start and end which is what we want.
    for (int i = 0; i <= n; i++) {
        curr_angle = i * angle_increment;
        glm::vec3 point(cos(curr_angle), sin(curr_angle), 0.0f);
        n_gon_points.push_back(point);
    }
    return n_gon_points;
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

std::vector<glm::vec3> generate_arrow_vertices(glm::vec2 start, glm::vec2 end, float stem_thickness, float tip_length) {

    float half_thickness = stem_thickness / (float)2;

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

    a_line = glm::normalize(a_line) * (float)sqrt(2) * tip_length;
    b_line = glm::normalize(b_line) * (float)sqrt(2) * tip_length;

    glm::vec2 g = end + a_line;
    glm::vec2 f = end + b_line;

    return {{a.x, a.y, 0.0f},     {b.x, b.y, 0.0f}, {c.x, c.y, 0.0f}, {d.x, d.y, 0.0f},
            {end.x, end.y, 0.0f}, {f.x, f.y, 0.0f}, {g.x, g.y, 0.0f}};
}

std::vector<unsigned int> generate_arrow_indices() { return {0, 1, 2, 2, 1, 3, 4, 5, 6}; }

// Function to scale vertices
void scale_vertices_in_place(std::vector<glm::vec3> &vertices, float scale_factor) {
    for (auto &vertex : vertices) {
        vertex *= scale_factor;
    }
}

void translate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &translation) {
    for (auto &vertex : vertices) {
        vertex += translation;
    }
}

void increment_indices_in_place(std::vector<unsigned int> &indices, unsigned int increase) {
    for (auto &index : indices) {
        index += increase;
    }
}
