#include "vertex_geometry.hpp"
#include <cassert>
#include <math.h>
#include "glm/geometric.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <numbers>

#include <vector>
#include <glm/vec3.hpp>

std::vector<Rectangle> generate_grid_rectangles(const glm::vec3 &center_position, float width, float height,
                                                int num_rectangles_x, int num_rectangles_y, float spacing) {
    std::vector<Rectangle> rectangles;

    float total_spacing_x = (num_rectangles_x - 1) * spacing;
    float total_spacing_y = (num_rectangles_y - 1) * spacing;

    float rectangle_width = (width - total_spacing_x) / num_rectangles_x;
    float rectangle_height = (height - total_spacing_y) / num_rectangles_y;

    float total_grid_width = width;
    float total_grid_height = height;

    float start_x = center_position.x - total_grid_width / 2;
    float start_y = center_position.y + total_grid_height / 2;

    for (int row = 0; row < num_rectangles_y; ++row) {
        for (int col = 0; col < num_rectangles_x; ++col) {
            float center_x = start_x + col * (rectangle_width + spacing) + rectangle_width / 2;
            float center_y = start_y - (row * (rectangle_height + spacing) + rectangle_height / 2);

            rectangles.emplace_back(
                Rectangle{glm::vec3(center_x, center_y, center_position.z), rectangle_width, rectangle_height});
        }
    }

    return rectangles;
}

IndexedVertices generate_grid(const glm::vec3 &center_position, float base_width, float base_height,
                              int num_rectangles_x, int num_rectangles_y, float spacing) {
    std::vector<Rectangle> rectangles =
        generate_grid_rectangles(center_position, base_width, base_height, num_rectangles_x, num_rectangles_y, spacing);

    std::vector<glm::vec3> vertices;
    std::vector<std::vector<unsigned int>> all_square_indices;

    for (const auto &rect : rectangles) {
        std::vector<glm::vec3> rectangle_vertices =
            generate_rectangle_vertices(rect.center.x, rect.center.y, rect.width, rect.height);
        vertices.insert(vertices.end(), rectangle_vertices.begin(), rectangle_vertices.end());

        std::vector<unsigned int> rectangle_indices = generate_rectangle_indices();
        all_square_indices.push_back(rectangle_indices);
    }

    std::vector<unsigned int> flattened_indices = flatten_and_increment_indices(all_square_indices);

    return {vertices, flattened_indices};
}

/**
 * @brief Flattens and increments a collection of index sets, ensuring that each set
 *        contains a contiguous range of integers {0, ..., n} for some integer n.
 *
 * This function concatenates multiple vectors of indices, incrementing each vector's
 * indices by the total number of vertices processed so far, ensuring that indices
 * are unique across all sets.
 *
 * @pre Each vector of indices must contain a contiguous set of integers {0, ..., n} for some integer n,
 *      though the indices do not have to be ordered.
 *
 * @param indices A vector of vectors, where each inner vector contains a contiguous set of integers.
 * @return A single vector of indices, flattened and adjusted to ensure uniqueness.
 *
 * @note The function asserts that each inner vector follows the precondition of having a contiguous set of integers.
 */
std::vector<unsigned int> flatten_and_increment_indices(const std::vector<std::vector<unsigned int>> &indices) {
    std::vector<unsigned int> flattened_indices;

    // Reserve space for all indices to avoid frequent reallocations
    size_t total_size = 0;
    for (const auto &inner_vec : indices) {
        total_size += inner_vec.size();
    }
    flattened_indices.reserve(total_size);

    unsigned int num_indices = 0; // Offset for ensuring unique indices

    // Flatten and increment indices
    for (const auto &inner_vec : indices) {
        if (!inner_vec.size()) {
            continue;
        }

        assert(!inner_vec.empty()); // Ensure the inner vector is non-empty

        // Find the maximum index in the current set
        unsigned int max_index = *std::max_element(inner_vec.begin(), inner_vec.end());

        // Create the expected set of contiguous indices {0, ..., max_index}
        std::set<unsigned int> expected_set;
        for (unsigned int i = 0; i <= max_index; ++i) {
            expected_set.insert(i);
        }

        // Create the actual set from the inner vector
        std::set<unsigned int> actual_set(inner_vec.begin(), inner_vec.end());

        // Assert that the actual set matches the expected contiguous set
        assert(actual_set == expected_set && "Indices do not form a contiguous set");

        // Increment indices and append to the flattened result
        for (unsigned int index : inner_vec) {
            flattened_indices.push_back(index + num_indices);
        }

        // Update the number of processed vertices (max_index + 1 = total vertices in this set)
        num_indices += max_index + 1;
    }

    return flattened_indices;
}

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

std::vector<glm::vec3> generate_rectangle_vertices_3d(const glm::vec3 &center, const glm::vec3 &width_dir,
                                                      const glm::vec3 &height_dir, float width, float height) {
    glm::vec3 half_width_vec = (width / 2.0f) * glm::normalize(width_dir);
    glm::vec3 half_height_vec = (height / 2.0f) * glm::normalize(height_dir);

    return {
        center + half_width_vec + half_height_vec, // top right
        center + half_width_vec - half_height_vec, // bottom right
        center - half_width_vec - half_height_vec, // bottom left
        center - half_width_vec + half_height_vec  // top left
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
 * @brief generate n equally spaced points on the unit circle
 *
 * Designed to work with glDrawArrays with a GL_TRIANGLE_FAN setting
 *
 * @TODO no longer needs to be called flattened.
 *
 */
std::vector<glm::vec3> generate_n_gon_flattened_vertices(int n) {
    assert(n >= 3);
    float angle_increment = (2 * std::numbers::pi) / (float)n;
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

std::vector<glm::vec3> generate_fibonacci_sphere_vertices(int num_samples, float scale) {
    std::vector<glm::vec3> points;
    float phi = M_PI * (std::sqrt(5.0) - 1.0);

    for (int i = 0; i < num_samples; i++) {
        float y = 1 - ((float)i / ((float)num_samples - 1)) * 2;
        float radius = std::sqrt(1 - y * y);
        float theta = phi * (float)i;

        float x = std::cos(theta) * radius;
        float z = std::sin(theta) * radius;
        points.emplace_back(x * scale, y * scale, z * scale);
    }

    return points;
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
