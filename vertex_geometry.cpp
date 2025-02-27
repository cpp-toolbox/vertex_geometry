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

#include <stdexcept>
#include <vector>

#include <map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vertex_geometry {

std::ostream &operator<<(std::ostream &os, const Rectangle &rect) {
    os << "Rectangle("
       << "Center: (" << rect.center.x << ", " << rect.center.y << ", " << rect.center.z << "), "
       << "Width: " << rect.width << ", "
       << "Height: " << rect.height << ")";
    return os;
}

float dot(const glm::vec3 &a, const glm::vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

bool is_right_angle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c) {
    glm::vec3 ab = b - a;
    glm::vec3 bc = c - b;
    return dot(ab, bc) == 0.0f; // Check if the vectors are perpendicular (dot product = 0)
}

bool are_valid_rectangle_corners(const glm::vec3 &top_left, const glm::vec3 &top_right, const glm::vec3 &bottom_left,
                                 const glm::vec3 &bottom_right) {
    return is_right_angle(top_left, top_right, bottom_right) && is_right_angle(top_right, bottom_right, bottom_left) &&
           is_right_angle(bottom_right, bottom_left, top_left) && is_right_angle(bottom_left, top_left, top_right);
}

Rectangle create_rectangle_from_corners(const glm::vec3 top_left, const glm::vec3 top_right,
                                        const glm::vec3 bottom_left, const glm::vec3 bottom_right) {
    // Check if the corners form a valid rectangle
    if (!are_valid_rectangle_corners(top_left, top_right, bottom_left, bottom_right)) {
        throw std::invalid_argument("The points do not form a valid rectangle.");
    }

    // Compute the center of the rectangle as the midpoint of the diagonals
    glm::vec3 diag1 = (top_left + bottom_right) / 2.0f;
    glm::vec3 diag2 = (top_right + bottom_left) / 2.0f;
    glm::vec3 center = (diag1 + diag2) / 2.0f;

    // Calculate the width and height using the distance between corners
    float width = glm::length(top_left - top_right);
    float height = glm::length(top_left - bottom_left);

    // Create and return the Rectangle object
    Rectangle rect;
    rect.center = center;
    rect.width = width;
    rect.height = height;

    return rect;
}

IndexedVertices Rectangle::get_ivs() const {
    return IndexedVertices(generate_rectangle_vertices(this->center.x, this->center.y, this->width, this->height),
                           generate_rectangle_indices());
}

// Constructor with explicit dimensions and origin
Grid::Grid(int rows, int cols, float width, float height, float origin_x, float origin_y)
    : rows(rows), cols(cols), grid_width(width), grid_height(height), origin_x(origin_x), origin_y(origin_y),
      rect_width(width / cols), rect_height(height / rows) {}

// Constructor using a Rectangle
Grid::Grid(int rows, int cols, const Rectangle &rect)
    : Grid(rows, cols, rect.width, rect.height, rect.center.x, rect.center.y) {}

// TODO: probably swap the order here?
Rectangle Grid::get_at(int col, int row) const {

    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Index out of range");
    }

    // Calculate the center position of the rectangle relative to the origin
    float x_center = origin_x - grid_width / 2 + rect_width * (col + 0.5f);   // X position
    float y_center = origin_y + grid_height / 2 - rect_height * (row + 0.5f); // Y position

    glm::vec3 center = {x_center, y_center, 0.0f}; // Center in 3D space

    return Rectangle{center, rect_width, rect_height};
}

std::vector<Rectangle> Grid::get_rectangles_in_bounding_box(int row1, int col1, int row2, int col2) const {
    if (row1 < 0 || row1 >= rows || row2 < 0 || row2 >= rows || col1 < 0 || col1 >= cols || col2 < 0 || col2 >= cols) {
        throw std::out_of_range("Row or column indices are out of bounds");
    }

    // Ensure row1, col1 is the top-left corner and row2, col2 is the bottom-right corner
    int top_row = std::min(row1, row2);
    int bottom_row = std::max(row1, row2);
    int left_col = std::min(col1, col2);
    int right_col = std::max(col1, col2);

    // Collect all rectangles in the bounding box
    std::vector<Rectangle> rectangles;
    for (int row = top_row; row <= bottom_row; ++row) {
        for (int col = left_col; col <= right_col; ++col) {
            rectangles.push_back(get_at(col, row));
        }
    }

    return rectangles;
}

// Get all rectangles in a specific row
std::vector<Rectangle> Grid::get_row(int row) const {
    if (row < 0 || row >= rows) {
        throw std::out_of_range("Row index out of range");
    }

    std::vector<Rectangle> row_rectangles;
    for (int col = 0; col < cols; ++col) {
        row_rectangles.push_back(get_at(col, row));
    }
    return row_rectangles;
}

// Get all rectangles in a specific column
std::vector<Rectangle> Grid::get_column(int col) const {
    if (col < 0 || col >= cols) {
        throw std::out_of_range("Column index out of range");
    }

    std::vector<Rectangle> col_rectangles;
    for (int row = 0; row < rows; ++row) {
        col_rectangles.push_back(get_at(col, row));
    }
    return col_rectangles;
}

Rectangle create_rectangle(float x_pos, float y_pos, float width, float height) {
    return {glm::vec3(x_pos, y_pos, 0), width, height};
}

Rectangle create_rectangle_from_top_left(const glm::vec3 &top_left, float width, float height) {
    return {top_left + glm::vec3(width / 2.0f, -height / 2.0f, 0.0f), width, height};
}

Rectangle create_rectangle_from_top_right(const glm::vec3 &top_right, float width, float height) {
    return {top_right + glm::vec3(-width / 2.0f, -height / 2.0f, 0.0f), width, height};
}

Rectangle create_rectangle_from_bottom_left(const glm::vec3 &bottom_left, float width, float height) {
    return {bottom_left + glm::vec3(width / 2.0f, height / 2.0f, 0.0f), width, height};
}

Rectangle create_rectangle_from_bottom_right(const glm::vec3 &bottom_right, float width, float height) {
    return {bottom_right + glm::vec3(-width / 2.0f, height / 2.0f, 0.0f), width, height};
}

Rectangle create_rectangle_from_center(const glm::vec3 &center, float width, float height) {
    return {center, width, height};
}

Rectangle expand_rectangle(const Rectangle &rect, float x_expand, float y_expand) {
    Rectangle expanded_rect;
    expanded_rect.center = rect.center;
    expanded_rect.width = rect.width + 2 * x_expand;
    expanded_rect.height = rect.height + 2 * y_expand;
    return expanded_rect;
}

Rectangle shrink_rectangle(const Rectangle &rect, float x_shrink, float y_shrink) {
    Rectangle shrunk_rect;
    shrunk_rect.center = rect.center;
    shrunk_rect.width = std::max(0.0f, rect.width - 2 * x_shrink);
    shrunk_rect.height = std::max(0.0f, rect.height - 2 * y_shrink);
    return shrunk_rect;
}

Rectangle slide_rectangle(const Rectangle &rect, int x_offset, int y_offset) {
    Rectangle new_rect = rect;

    // Slide the rectangle's center by the given offsets multiplied by width and height
    new_rect.center.x += x_offset * rect.width;
    new_rect.center.y += y_offset * rect.height;

    return new_rect;
}

Rectangle get_bounding_rectangle(const std::vector<Rectangle> &rectangles) {
    if (rectangles.empty()) {
        return {{0.0f, 0.0f, 0.0f}, 0.0f, 0.0f};
    }

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto &rect : rectangles) {
        float left = rect.center.x - rect.width / 2;
        float right = rect.center.x + rect.width / 2;
        float bottom = rect.center.y - rect.height / 2;
        float top = rect.center.y + rect.height / 2;

        if (left < min_x)
            min_x = left;
        if (right > max_x)
            max_x = right;
        if (bottom < min_y)
            min_y = bottom;
        if (top > max_y)
            max_y = top;
    }

    float bounding_width = max_x - min_x;
    float bounding_height = max_y - min_y;
    glm::vec3 bounding_center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f, 0.0f);

    return {bounding_center, bounding_width, bounding_height};
}

std::vector<Rectangle> weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights,
                                            bool vertical) {
    std::vector<Rectangle> subrectangles;
    float total_weight = 0.0f;

    // Calculate total weight
    for (auto weight : weights) {
        total_weight += static_cast<float>(weight);
    }

    // Initialize start position (for top-left corner or bottom-left corner)
    float start_position = (vertical) ? rect.center.y + rect.height / 2 : rect.center.x - rect.width / 2;
    float current_position = start_position;

    // Generate subrectangles
    for (size_t i = 0; i < weights.size(); ++i) {
        float subdivision_size =
            (static_cast<float>(weights[i]) / total_weight) * (vertical ? rect.height : rect.width);

        Rectangle subrect;
        if (vertical) {
            // Create subrectangle vertically from top to bottom
            subrect.center = rect.center;
            subrect.center.y = current_position - subdivision_size / 2;
            subrect.width = rect.width;
            subrect.height = subdivision_size;
            // Update the current position for the next subdivision (move downwards)
            current_position -= subdivision_size;
        } else {
            // Create subrectangle horizontally from left to right
            subrect.center = rect.center;
            subrect.center.x = current_position + subdivision_size / 2;
            subrect.width = subdivision_size;
            subrect.height = rect.height;
            // Update the current position for the next subdivision (move rightwards)
            current_position += subdivision_size;
        }

        subrectangles.push_back(subrect);
    }

    return subrectangles;
}

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
 * @brief Flattens and increments a collection of index sets
 *
 *
 * @pre Each vector of indices must contain a contiguous set of integers {0, ..., n} for some integer n,
 *      though the indices do not have to be ordered.
 *
 * This function concatenates multiple vectors of indices, incrementing each vector's
 * indices by the total number of vertices processed so far, ensuring that indices
 * are unique across all sets.
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

draw_info::IndexedVertexPositions generate_rectangle(float center_x, float center_y, float width, float height) {
    return {generate_rectangle_indices(), generate_rectangle_vertices(center_x, center_y, width, height)};
}
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

draw_info::IndexedVertexPositions generate_cone(int segments, float height, float radius) {
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    float half_height = height / 2.0f;
    float angle_increment = 2.0f * M_PI / segments;

    // Top vertex (apex of the cone)
    vertices.push_back(glm::vec3(0.0f, half_height, 0.0f));

    // Bottom center vertex
    vertices.push_back(glm::vec3(0.0f, -half_height, 0.0f));

    // Generate vertices for the bottom circle
    for (int i = 0; i < segments; ++i) {
        float angle = i * angle_increment;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        vertices.push_back(glm::vec3(x, -half_height, z));
    }

    // Generate indices for the bottom face (fan)
    for (int i = 0; i < segments; ++i) {
        indices.push_back(1); // Bottom center vertex
        indices.push_back(2 + i);
        indices.push_back(2 + ((i + 1) % segments));
    }

    // Generate indices for the side faces
    for (int i = 0; i < segments; ++i) {
        indices.push_back(0); // Apex of the cone
        indices.push_back(2 + ((i + 1) % segments));
        indices.push_back(2 + i);
    }

    return {indices, vertices};
}

draw_info::IndexedVertexPositions generate_cylinder(int segments, float height, float radius) {
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    float half_height = height / 2.0f;
    float angle_increment = 2.0f * M_PI / segments;

    // Top center vertex
    vertices.push_back(glm::vec3(0.0f, half_height, 0.0f));

    // Bottom center vertex
    vertices.push_back(glm::vec3(0.0f, -half_height, 0.0f));

    // Generate vertices for top and bottom circles
    for (int i = 0; i < segments; ++i) {
        float angle = i * angle_increment;
        float x = radius * cos(angle);
        float z = radius * sin(angle);

        // Top circle
        vertices.push_back(glm::vec3(x, half_height, z));

        // Bottom circle
        vertices.push_back(glm::vec3(x, -half_height, z));
    }

    // Generate indices for the top face
    for (int i = 0; i < segments; ++i) {
        indices.push_back(0); // Top center vertex
        indices.push_back(2 + (i * 2));
        indices.push_back(2 + ((i * 2 + 2) % (segments * 2)));
    }

    // Generate indices for the bottom face
    for (int i = 0; i < segments; ++i) {
        indices.push_back(1); // Bottom center vertex
        indices.push_back(3 + ((i * 2) % (segments * 2)));
        indices.push_back(3 + ((i * 2 + 2) % (segments * 2)));
    }

    // Generate indices for the side faces
    for (int i = 0; i < segments; ++i) {
        int top1 = 2 + (i * 2);
        int bottom1 = 3 + (i * 2);
        int top2 = 2 + ((i * 2 + 2) % (segments * 2));
        int bottom2 = 3 + ((i * 2 + 2) % (segments * 2));

        // First triangle
        indices.push_back(top1);
        indices.push_back(bottom1);
        indices.push_back(top2);

        // Second triangle
        indices.push_back(top2);
        indices.push_back(bottom1);
        indices.push_back(bottom2);
    }

    return {indices, vertices};
}

// Function to generate the initial icosahedron vertices
std::vector<glm::vec3> generate_initial_icosahedron_vertices(float radius) {
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
    return {glm::normalize(glm::vec3(-1, phi, 0)) * radius,  glm::normalize(glm::vec3(1, phi, 0)) * radius,
            glm::normalize(glm::vec3(-1, -phi, 0)) * radius, glm::normalize(glm::vec3(1, -phi, 0)) * radius,
            glm::normalize(glm::vec3(0, -1, phi)) * radius,  glm::normalize(glm::vec3(0, 1, phi)) * radius,
            glm::normalize(glm::vec3(0, -1, -phi)) * radius, glm::normalize(glm::vec3(0, 1, -phi)) * radius,
            glm::normalize(glm::vec3(phi, 0, -1)) * radius,  glm::normalize(glm::vec3(phi, 0, 1)) * radius,
            glm::normalize(glm::vec3(-phi, 0, -1)) * radius, glm::normalize(glm::vec3(-phi, 0, 1)) * radius};
}

// Function to generate the initial icosahedron indices
std::vector<unsigned int> generate_initial_icosahedron_indices() {
    return {0u, 11u, 5u,  0u, 5u,  1u, 0u, 1u, 7u, 0u, 7u,  10u, 0u, 10u, 11u, 1u, 5u, 9u, 5u, 11u,
            4u, 11u, 10u, 2u, 10u, 7u, 6u, 7u, 1u, 8u, 3u,  9u,  4u, 3u,  4u,  2u, 3u, 2u, 6u, 3u,
            6u, 8u,  3u,  8u, 9u,  4u, 9u, 5u, 2u, 4u, 11u, 6u,  2u, 10u, 8u,  6u, 7u, 9u, 8u, 1u};
}

// Function to get the midpoint vertex index
int get_midpoint(int a, int b, std::map<std::pair<int, int>, int> &midpoint_cache, std::vector<glm::vec3> &vertices,
                 float radius) {
    auto key = std::minmax(a, b);
    if (midpoint_cache.count(key)) {
        return midpoint_cache[key];
    }
    glm::vec3 mid = glm::normalize((vertices[a] + vertices[b]) * 0.5f) * radius;
    vertices.push_back(mid);
    int idx = vertices.size() - 1;
    midpoint_cache[key] = idx;
    return idx;
}

// Function to subdivide the icosahedron
void subdivide_icosahedron(int subdivisions, std::vector<glm::vec3> &vertices, std::vector<unsigned int> &indices,
                           float radius) {
    std::map<std::pair<int, int>, int> midpoint_cache;
    for (int i = 0; i < subdivisions; ++i) {
        std::vector<unsigned int> new_indices;
        for (size_t j = 0; j < indices.size(); j += 3) {
            int a = indices[j];
            int b = indices[j + 1];
            int c = indices[j + 2];

            int ab = get_midpoint(a, b, midpoint_cache, vertices, radius);
            int bc = get_midpoint(b, c, midpoint_cache, vertices, radius);
            int ca = get_midpoint(c, a, midpoint_cache, vertices, radius);

            new_indices.insert(new_indices.end(), {static_cast<unsigned int>(a), static_cast<unsigned int>(ab),
                                                   static_cast<unsigned int>(ca), static_cast<unsigned int>(b),
                                                   static_cast<unsigned int>(bc), static_cast<unsigned int>(ab),
                                                   static_cast<unsigned int>(c), static_cast<unsigned int>(ca),
                                                   static_cast<unsigned int>(bc), static_cast<unsigned int>(ab),
                                                   static_cast<unsigned int>(bc), static_cast<unsigned int>(ca)});
        }
        indices = std::move(new_indices);
    }
}

draw_info::IndexedVertexPositions generate_icosphere(int subdivisions, float radius) {
    std::vector<glm::vec3> vertices = generate_initial_icosahedron_vertices(radius);
    std::vector<unsigned int> indices = generate_initial_icosahedron_indices();

    subdivide_icosahedron(subdivisions, vertices, indices, radius);

    return {indices, vertices};
}

void merge_ivps(draw_info::IndexedVertexPositions &base_ivp, const draw_info::IndexedVertexPositions &extend_ivp) {
    unsigned int base_vertex_count = base_ivp.xyz_positions.size();

    base_ivp.xyz_positions.insert(base_ivp.xyz_positions.end(), extend_ivp.xyz_positions.begin(),
                                  extend_ivp.xyz_positions.end());

    for (unsigned int index : extend_ivp.indices) {
        base_ivp.indices.push_back(index + base_vertex_count);
    }
}

void merge_ivps(draw_info::IndexedVertexPositions &base_ivp,
                const std::vector<draw_info::IndexedVertexPositions> &extend_ivps) {
    for (const auto &extend_ivp : extend_ivps) {
        merge_ivps(base_ivp, extend_ivp);
    }
}

// std::vector<unsigned int> flatten_and_increment_indices(const std::vector<std::vector<unsigned int>> &indices) {

draw_info::IndexedVertexPositions generate_unit_cube() {
    return {generate_cube_indices(), generate_unit_cube_vertices()};
}

std::vector<glm::vec3> cube_vertex_positions = {{-1.0f, -1.0f, 1.0f},  // 0  Coordinates
                                                {1.0f, -1.0f, 1.0f},   // 1        7--------6
                                                {1.0f, -1.0f, -1.0f},  // 2       /|       /|
                                                {-1.0f, -1.0f, -1.0f}, // 3      4--------5 |
                                                {-1.0f, 1.0f, 1.0f},   // 4      | |      | |
                                                {1.0f, 1.0f, 1.0f},    // 5      | 3------|-2
                                                {1.0f, 1.0f, -1.0f},   // 6      |/       |/
                                                {-1.0f, 1.0f, -1.0f}}; // 7      0--------1

std::vector<glm::vec3> generate_unit_cube_vertices() { return cube_vertex_positions; }

std::vector<unsigned int> cube_vertex_indices = { // Right
    1, 2, 6, 6, 5, 1,
    // Left
    0, 4, 7, 7, 3, 0,
    // Top
    4, 5, 6, 6, 7, 4,
    // Bottom
    0, 3, 2, 2, 1, 0,
    // Back
    0, 1, 5, 5, 4, 0,
    // Front
    3, 7, 6, 6, 2, 3};

std::vector<unsigned int> generate_cube_indices() { return cube_vertex_indices; }

/**
 * @brief Generates normals for each corner of a rectangle.
 *
 * The rectangle is assumed to lie in the XY plane, with normals pointing along the positive Z-axis.
 *
 * @return A vector of glm::vec3 representing the normals for each corner.
 */
std::vector<glm::vec3> generate_rectangle_normals() {
    return {
        {0.0f, 0.0f, 1.0f}, // Normal for top right corner
        {0.0f, 0.0f, 1.0f}, // Normal for bottom right corner
        {0.0f, 0.0f, 1.0f}, // Normal for bottom left corner
        {0.0f, 0.0f, 1.0f}  // Normal for top left corner
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

std::vector<glm::vec2> generate_rectangle_texture_coordinates() {
    // note that the order is reversed as compred to the vertices becamse
    // we load in our images in the regular top left origin orientation
    // and then opengl uses a bottom left origin, so we account for that here
    return {
        glm::vec2(1.0f, 0.0f), // Bottom-right
        glm::vec2(1.0f, 1.0f), // Top-right
        glm::vec2(0.0f, 1.0f), // Top-left
        glm::vec2(0.0f, 0.0f)  // Bottom-left
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

draw_info::IndexedVertexPositions generate_annulus(float center_x, float center_y, float outer_radius,
                                                   float inner_radius, int num_segments, float percent) {
    return {generate_annulus_indices(num_segments, percent),
            generate_annulus_vertices(center_x, center_y, outer_radius, inner_radius, num_segments, percent)};
}

std::vector<glm::vec3> generate_annulus_vertices(float center_x, float center_y, float outer_radius, float inner_radius,
                                                 int num_segments, float percent) {
    std::vector<glm::vec3> vertices;
    float full_circle = 2.0f * M_PI;
    float angle_step = full_circle / num_segments;

    num_segments *= percent;

    for (int i = 0; i < num_segments; ++i) {
        float angle = i * angle_step;

        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);

        // outer ring
        vertices.emplace_back(center_x + outer_radius * cos_angle, center_y + outer_radius * sin_angle, 0.0f);
        // inner ring
        vertices.emplace_back(center_x + inner_radius * cos_angle, center_y + inner_radius * sin_angle, 0.0f);
    }

    return vertices;
}

std::vector<unsigned int> generate_annulus_indices(int num_segments, float percent) {
    std::vector<unsigned int> indices;

    num_segments *= percent;

    for (int i = 0; i < num_segments; ++i) {
        int next = (i + 1) % num_segments;

        // triangle 1
        indices.push_back(2 * i);
        indices.push_back(2 * next);
        indices.push_back(2 * i + 1);

        // triangle 2
        indices.push_back(2 * next);
        indices.push_back(2 * next + 1);
        indices.push_back(2 * i + 1);
    }

    return indices;
}

std::vector<glm::vec3> generate_star_vertices(float center_x, float center_y, float outer_radius, float inner_radius,
                                              int num_star_tips, bool blunt_tips) {

    int star_multiplier = (blunt_tips ? 3 : 2);
    int num_vertices_required = star_multiplier * num_star_tips;

    int inner_radius_offset = 1;

    float full_rotation = 2 * M_PI;

    std::vector<glm::vec3> vertices;
    float angle_step = full_rotation / num_vertices_required;

    // make sure that a tip always faces directly up
    float initial_angle = full_rotation / 4;

    // align the flat part of the blunt tip so that it points vertically
    if (blunt_tips) {
        initial_angle -= angle_step / 2;
    }

    for (int i = 0; i < num_vertices_required; i++) {
        float angle = initial_angle + i * angle_step;

        float radius;
        if ((i + inner_radius_offset) % star_multiplier == 0) {
            radius = inner_radius;
        } else {
            radius = outer_radius;
        }

        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);

        vertices.emplace_back(center_x + radius * cos_angle, center_y + radius * sin_angle, 0.0f);
    }

    // the center vertex is required to form triangles and is stored at the very end.
    vertices.emplace_back(center_x, center_y, 0);

    return vertices;
}

std::vector<unsigned int> generate_star_indices(int num_star_tips, bool blunt_tips) {
    std::vector<unsigned int> indices;

    int star_multiplier = (blunt_tips ? 3 : 2);
    int num_vertices = star_multiplier * num_star_tips;

    for (int i = 0; i < num_vertices; ++i) {
        int next = (i + 1) % num_vertices;
        indices.push_back(i);
        indices.push_back(next);
        indices.push_back(num_vertices); // center index as specified in above function
    }

    return indices;
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
} // namespace vertex_geometry
