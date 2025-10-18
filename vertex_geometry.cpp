#include "vertex_geometry.hpp"
#include <cassert>
#include <functional>
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

std::string strip_leading_newlines(const std::string &text) {
    size_t start = 0;
    while (start < text.size() && text[start] == '\n') {
        ++start;
    }
    return text.substr(start);
}

bool circle_intersects_rect(float cx, float cy, float radius, const Rectangle &rect) {
    // Clamp circle center to rectangle bounds to find the closest point on the rect
    float closest_x = std::max(rect.center.x, std::min(cx, rect.center.x + rect.width));
    float closest_y = std::max(rect.center.y, std::min(cy, rect.center.y + rect.height));

    float dx = closest_x - cx;
    float dy = closest_y - cy;

    return dx * dx + dy * dy <= radius * radius;
}

std::vector<Rectangle> get_rects_intersecting_circle(const Grid &grid, float cx, float cy, float radius) {
    float x0 = cx - radius;
    float y0 = cy - radius;
    float x1 = cx + radius;
    float y1 = cy + radius;

    std::vector<Rectangle> candidates = grid.get_selection(x0, y0, x1, y1);
    std::vector<Rectangle> result;

    for (const Rectangle &rect : candidates) {
        if (circle_intersects_rect(cx, cy, radius, rect)) {
            result.push_back(rect);
        }
    }

    return result;
}

draw_info::IndexedVertexPositions text_grid_to_rect_grid(const std::string &text_grid,
                                                         const vertex_geometry::Rectangle bounding_rect) {
    unsigned int rows = 0;
    unsigned int cols = 0;

    // count rows and columns based on text_grid.
    std::vector<std::string> lines;
    std::string line;
    std::string cleaned_text_grid = strip_leading_newlines(text_grid);

    for (char c : cleaned_text_grid) {
        if (c == '\n') {
            lines.push_back(line);
            line.clear();
        } else {
            line += c;
        }
    }
    if (!line.empty())
        lines.push_back(line); // for the last line if there's no final newline.

    rows = lines.size();
    if (rows > 0) {
        cols = lines[0].length(); // assuming all rows have equal length.
    }

    // Initialize grid
    vertex_geometry::Grid grid(rows, cols, bounding_rect);

    std::vector<draw_info::IndexedVertexPositions> ivps;

    // iterate over the grid and collect indexed vertex positions for '*' characters.
    for (unsigned int row = 0; row < rows; ++row) {
        for (unsigned int col = 0; col < cols; ++col) {
            if (lines[row][col] == '*') {
                vertex_geometry::Rectangle rect = grid.get_at(col, row);
                draw_info::IndexedVertexPositions ivp = rect.get_ivs();
                ivps.push_back(ivp);
            }
        }
    }

    return vertex_geometry::merge_ivps(ivps);
}

draw_info::IndexedVertexPositions generate_rectangle_between_2d(const glm::vec2 &p1, const glm::vec2 &p2,
                                                                float thickness) {
    glm::vec2 dir = p2 - p1;
    float length = glm::length(dir);
    if (length == 0.0f) {
        throw std::invalid_argument("Points must not be identical.");
    }

    glm::vec2 dir_norm = glm::normalize(dir);
    glm::vec2 perp = glm::vec2(-dir_norm.y, dir_norm.x) * (thickness / 2.0f);

    // Convert 2D points to 3D with z = 0
    glm::vec3 p1_left = glm::vec3(p1 - perp, 0.0f);
    glm::vec3 p1_right = glm::vec3(p1 + perp, 0.0f);
    glm::vec3 p2_left = glm::vec3(p2 - perp, 0.0f);
    glm::vec3 p2_right = glm::vec3(p2 + perp, 0.0f);

    std::vector<glm::vec3> vertices = {
        p1_left,  // 0
        p1_right, // 1
        p2_right, // 2
        p2_left   // 3
    };

    std::vector<unsigned int> indices = {0, 1, 2, 2, 3, 0};

    return {indices, vertices};
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

draw_info::IndexedVertexPositions Rectangle::get_ivs() const {
    return {
        generate_rectangle_indices(),
        generate_rectangle_vertices_with_z(this->center.x, this->center.y, this->center.z, this->width, this->height)};
}

glm::vec3 Rectangle::get_top_left() const { return center + glm::vec3(-width / 2.0f, height / 2.0f, 0.0f); }
glm::vec3 Rectangle::get_top_center() const { return center + glm::vec3(0.0f, height / 2.0f, 0.0f); }
glm::vec3 Rectangle::get_top_right() const { return center + glm::vec3(width / 2.0f, height / 2.0f, 0.0f); }
glm::vec3 Rectangle::get_center_left() const { return center + glm::vec3(-width / 2.0f, 0.0f, 0.0f); }
glm::vec3 Rectangle::get_center_right() const { return center + glm::vec3(width / 2.0f, 0.0f, 0.0f); }
glm::vec3 Rectangle::get_bottom_left() const { return center + glm::vec3(-width / 2.0f, -height / 2.0f, 0.0f); }
glm::vec3 Rectangle::get_bottom_center() const { return center + glm::vec3(0.0f, -height / 2.0f, 0.0f); }
glm::vec3 Rectangle::get_bottom_right() const { return center + glm::vec3(width / 2.0f, -height / 2.0f, 0.0f); }

// Constructor with explicit dimensions and origin
Grid::Grid(int rows, int cols, float width, float height, float origin_x, float origin_y, float origin_z)
    : rows(rows), cols(cols), grid_width(width), grid_height(height), origin_x(origin_x), origin_y(origin_y),
      origin_z(origin_z), rect_width(width / cols), rect_height(height / rows) {}

// Constructor using a Rectangle
Grid::Grid(int rows, int cols, const Rectangle &rect)
    : Grid(rows, cols, rect.width, rect.height, rect.center.x, rect.center.y, rect.center.z) {}

// TODO: probably swap the order here?
Rectangle Grid::get_at(int col, int row) const {

    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Index out of range");
    }

    // Calculate the center position of the rectangle relative to the origin
    float x_center = origin_x - grid_width / 2 + rect_width * (col + 0.5f);   // X position
    float y_center = origin_y + grid_height / 2 - rect_height * (row + 0.5f); // Y position

    glm::vec3 center = {x_center, y_center, origin_z}; // Center in 3D space

    return Rectangle{center, rect_width, rect_height};
}

std::vector<Rectangle> Grid::get_selection(float x0, float y0, float x1, float y1) const {
    // Normalize the coordinates (x0, y0) should be bottom-left, (x1, y1) top-right
    float min_x = std::min(x0, x1);
    float max_x = std::max(x0, x1);
    float min_y = std::min(y0, y1);
    float max_y = std::max(y0, y1);

    // Calculate the bounding columns and rows
    int start_col = static_cast<int>(std::floor((min_x - (origin_x - grid_width / 2)) / rect_width));
    int end_col = static_cast<int>(std::floor((max_x - (origin_x - grid_width / 2)) / rect_width));

    int start_row = static_cast<int>(std::floor(((origin_y + grid_height / 2) - max_y) / rect_height));
    int end_row = static_cast<int>(std::floor(((origin_y + grid_height / 2) - min_y) / rect_height));

    std::vector<Rectangle> selected;

    for (int row = start_row; row <= end_row; ++row) {
        for (int col = start_col; col <= end_col; ++col) {
            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                selected.push_back(get_at(col, row));
            }
        }
    }

    return selected;
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

Rectangle create_rectangle_from_center_left(const glm::vec3 &center_left, float width, float height) {
    return {center_left + glm::vec3(width / 2.0f, 0.0f, 0.0f), width, height};
}

Rectangle create_rectangle_from_center(const glm::vec3 &center, float width, float height) {
    return {center, width, height};
}

draw_info::IndexedVertexPositions AxisAlignedBoundingBox::get_ivp() {

    auto corners = get_corners();

    // Bottom face (z = min.z)
    std::vector<glm::vec3> bottom_pts = {
        corners[0], // (min.x, min.y, min.z)
        corners[1], // (max.x, min.y, min.z)
        corners[3], // (max.x, max.y, min.z)
        corners[2]  // (min.x, max.y, min.z)
    };

    // Top face (z = max.z)
    std::vector<glm::vec3> top_pts = {
        corners[4], // (min.x, min.y, max.z)
        corners[5], // (max.x, min.y, max.z)
        corners[7], // (max.x, max.y, max.z)
        corners[6]  // (min.x, max.y, max.z)
    };

    NGon bottom_ngon(bottom_pts, TriangulationMode::VertexFan);
    NGon top_ngon(top_pts, TriangulationMode::VertexFan);

    // Connect ngons into a prism IVP
    return connect_ngons(bottom_ngon, top_ngon);
}

draw_info::IndexedVertexPositions triangulate_ngon(const NGon &ngon) {
    const auto &pts = ngon.get_points();
    std::vector<glm::vec3> xyz_positions(pts.begin(), pts.end());
    std::vector<unsigned int> indices;

    if (ngon.get_triangulation_mode() == TriangulationMode::CentralFan) {
        glm::vec3 centroid(0.0f);
        for (const auto &p : pts)
            centroid += p;
        centroid /= static_cast<float>(pts.size());

        unsigned int center_index = static_cast<unsigned int>(xyz_positions.size());
        xyz_positions.push_back(centroid);

        for (std::size_t i = 0; i < pts.size(); ++i) {
            unsigned int next = (i + 1) % pts.size();
            indices.push_back(center_index);
            indices.push_back(static_cast<unsigned int>(i));
            indices.push_back(static_cast<unsigned int>(next));
        }
    } else { // VertexFan
        for (std::size_t i = 1; i < pts.size() - 1; ++i) {
            indices.push_back(0);
            indices.push_back(static_cast<unsigned int>(i));
            indices.push_back(static_cast<unsigned int>(i + 1));
        }
    }

    return draw_info::IndexedVertexPositions(indices, xyz_positions);
}

// Connect two Ngons dynamically
draw_info::IndexedVertexPositions connect_ngons(const NGon &a, const NGon &b) {
    auto capA = triangulate_ngon(a);
    auto capB = triangulate_ngon(b);

    std::vector<glm::vec3> xyz_positions;
    xyz_positions.reserve(capA.xyz_positions.size() + capB.xyz_positions.size());
    xyz_positions.insert(xyz_positions.end(), capA.xyz_positions.begin(), capA.xyz_positions.end());
    xyz_positions.insert(xyz_positions.end(), capB.xyz_positions.begin(), capB.xyz_positions.end());

    std::vector<unsigned int> indices;
    indices.insert(indices.end(), capA.indices.begin(), capA.indices.end());

    unsigned int offsetB = static_cast<unsigned int>(capA.xyz_positions.size());
    for (unsigned int idx : capB.indices)
        indices.push_back(offsetB + idx);

    // Sides
    std::size_t N = a.size();
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t next = (i + 1) % N;
        unsigned int a0 = static_cast<unsigned int>(i);
        unsigned int a1 = static_cast<unsigned int>(next);
        unsigned int b0 = static_cast<unsigned int>(offsetB + i);
        unsigned int b1 = static_cast<unsigned int>(offsetB + next);

        indices.push_back(a0);
        indices.push_back(a1);
        indices.push_back(b1);
        indices.push_back(a0);
        indices.push_back(b1);
        indices.push_back(b0);
    }

    return draw_info::IndexedVertexPositions(indices, xyz_positions);
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

Rectangle scale_rectangle_from_left_side(const Rectangle &rect, float x_shrink, float y_shrink) {
    return create_rectangle_from_center_left(rect.get_center_left(), rect.width * x_shrink, rect.height * y_shrink);
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

std::vector<Rectangle> subdivide_rectangle(const Rectangle &rect, unsigned int num_subdivisions, bool vertical) {
    std::vector<unsigned int> even_weights(num_subdivisions, 1);
    return weighted_subdivision(rect, even_weights, vertical);
}

std::vector<Rectangle> vertical_weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights) {
    return weighted_subdivision(rect, weights);
}

std::vector<Rectangle> horizontal_weighted_subdivision(const Rectangle &rect,
                                                       const std::vector<unsigned int> &weights) {
    return weighted_subdivision(rect, weights, false);
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

draw_info::IndexedVertexPositions generate_grid(const glm::vec3 &center_position, float base_width, float base_height,
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

    return {flattened_indices, vertices};
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
 * @note The function asserts that each inner vector follows the precondition of having a contiguous set of
 * integers.
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

draw_info::IndexedVertexPositions generate_rectangle(float center_x, float center_y, float center_z, float width,
                                                     float height) {
    return {generate_rectangle_indices(),
            generate_rectangle_vertices_with_z(center_x, center_y, center_z, width, height)};
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

std::vector<glm::vec3> generate_rectangle_vertices_with_z(float center_x, float center_y, float center_z, float width,
                                                          float height) {

    float half_width = width / (float)2;
    float half_height = height / (float)2;

    // todo currently makes a rectangle twice as big as side length
    return {
        {center_x + half_width, center_y + half_height, center_z}, // top right
        {center_x + half_width, center_y - half_height, center_z}, // bottom right
        {center_x - half_width, center_y - half_height, center_z}, // bottom left
        {center_x - half_width, center_y + half_height, center_z}  // top left
    };
}

draw_info::IndexedVertexPositions generate_cone_between(const glm::vec3 &base, const glm::vec3 &tip, int segments,
                                                        float radius) {
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    // compute the axis vector and length
    glm::vec3 axis = tip - base;
    float height = glm::length(axis);
    glm::vec3 dir = glm::normalize(axis);

    // find a perpendicular vector to construct the base circle
    glm::vec3 up = (fabs(dir.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 tangent = glm::normalize(glm::cross(dir, up));
    glm::vec3 bitangent = glm::cross(dir, tangent);

    // tip and base center
    vertices.push_back(tip);  // tip vertex
    vertices.push_back(base); // base center vertex

    // generate base circle vertices
    float angle_increment = 2.0f * std::numbers::pi / segments;
    for (int i = 0; i < segments; ++i) {
        float angle = i * angle_increment;
        glm::vec3 offset = radius * (cos(angle) * tangent + sin(angle) * bitangent);
        vertices.push_back(base + offset);
    }

    // generate indices for the base (fan)
    for (int i = 0; i < segments; ++i) {
        indices.push_back(1); // base center
        indices.push_back(2 + ((i + 1) % segments));
        indices.push_back(2 + i);
    }

    // generate indices for the side faces
    for (int i = 0; i < segments; ++i) {
        indices.push_back(0); // tip
        indices.push_back(2 + i);
        indices.push_back(2 + ((i + 1) % segments));
    }

    return {indices, vertices};
}

// about to do torus

draw_info::IVPNormals generate_torus(int major_segments, int minor_segments, float major_radius, float minor_radius) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;

    for (int i = 0; i < major_segments; ++i) {
        float major_angle = 2.0f * std::numbers::pi * i / major_segments;
        glm::vec3 circle_center = glm::vec3(cos(major_angle), 0.0f, sin(major_angle)) * major_radius;

        glm::vec3 tangent = glm::vec3(-sin(major_angle), 0.0f, cos(major_angle)); // local X
        glm::vec3 bitangent = glm::vec3(0.0f, 1.0f, 0.0f);                        // local Y
        glm::vec3 normal = glm::normalize(glm::cross(tangent, bitangent));        // local Z

        for (int j = 0; j < minor_segments; ++j) {
            float minor_angle = 2.0f * std::numbers::pi * j / minor_segments;
            float x = cos(minor_angle);
            float y = sin(minor_angle);

            glm::vec3 local_offset = normal * x * minor_radius + bitangent * y * minor_radius;
            glm::vec3 position = circle_center + local_offset;
            glm::vec3 vertex_normal = glm::normalize(local_offset); // normal points away from tube center

            vertices.push_back(position);
            normals.push_back(vertex_normal);
        }
    }

    for (int i = 0; i < major_segments; ++i) {
        for (int j = 0; j < minor_segments; ++j) {
            int next_i = (i + 1) % major_segments;
            int next_j = (j + 1) % minor_segments;

            int current = i * minor_segments + j;
            int right = i * minor_segments + next_j;
            int below = next_i * minor_segments + j;
            int below_right = next_i * minor_segments + next_j;

            // First triangle
            indices.push_back(current);
            indices.push_back(below_right);
            indices.push_back(right);

            // Second triangle
            indices.push_back(current);
            indices.push_back(below);
            indices.push_back(below_right);
        }
    }

    return draw_info::IVPNormals(indices, vertices, normals);
}

draw_info::IVPNormals generate_cube(float size) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;

    float h = size / 2.0f;

    // Define the 6 cube faces with their normals
    struct Face {
        glm::vec3 normal;
        glm::vec3 corners[4];
    };

    std::vector<Face> faces = {
        // Front (+Z)
        {glm::vec3(0, 0, 1),
         {
             glm::vec3(-h, -h, h),
             glm::vec3(h, -h, h),
             glm::vec3(h, h, h),
             glm::vec3(-h, h, h),
         }},
        // Back (-Z)
        {glm::vec3(0, 0, -1),
         {
             glm::vec3(h, -h, -h),
             glm::vec3(-h, -h, -h),
             glm::vec3(-h, h, -h),
             glm::vec3(h, h, -h),
         }},
        // Left (-X)
        {glm::vec3(-1, 0, 0),
         {
             glm::vec3(-h, -h, -h),
             glm::vec3(-h, -h, h),
             glm::vec3(-h, h, h),
             glm::vec3(-h, h, -h),
         }},
        // Right (+X)
        {glm::vec3(1, 0, 0),
         {
             glm::vec3(h, -h, h),
             glm::vec3(h, -h, -h),
             glm::vec3(h, h, -h),
             glm::vec3(h, h, h),
         }},
        // Top (+Y)
        {glm::vec3(0, 1, 0),
         {
             glm::vec3(-h, h, h),
             glm::vec3(h, h, h),
             glm::vec3(h, h, -h),
             glm::vec3(-h, h, -h),
         }},
        // Bottom (-Y)
        {glm::vec3(0, -1, 0),
         {
             glm::vec3(-h, -h, -h),
             glm::vec3(h, -h, -h),
             glm::vec3(h, -h, h),
             glm::vec3(-h, -h, h),
         }},
    };

    for (const auto &face : faces) {
        unsigned int start_index = static_cast<unsigned int>(vertices.size());
        for (int i = 0; i < 4; ++i) {
            vertices.push_back(face.corners[i]);
            normals.push_back(face.normal);
        }
        indices.insert(indices.end(),
                       {start_index, start_index + 1, start_index + 2, start_index, start_index + 2, start_index + 3});
    }

    return draw_info::IVPNormals(std::move(indices), std::move(vertices), std::move(normals));
}

draw_info::IVPNormals generate_box(float size_x, float size_y, float size_z) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;

    float hx = size_x / 2.0f;
    float hy = size_y / 2.0f;
    float hz = size_z / 2.0f;

    struct Face {
        glm::vec3 normal;
        glm::vec3 corners[4];
    };

    std::vector<Face> faces = {
        // Front (+Z)
        {glm::vec3(0, 0, 1),
         {
             glm::vec3(-hx, -hy, hz),
             glm::vec3(hx, -hy, hz),
             glm::vec3(hx, hy, hz),
             glm::vec3(-hx, hy, hz),
         }},
        // Back (-Z)
        {glm::vec3(0, 0, -1),
         {
             glm::vec3(hx, -hy, -hz),
             glm::vec3(-hx, -hy, -hz),
             glm::vec3(-hx, hy, -hz),
             glm::vec3(hx, hy, -hz),
         }},
        // Left (-X)
        {glm::vec3(-1, 0, 0),
         {
             glm::vec3(-hx, -hy, -hz),
             glm::vec3(-hx, -hy, hz),
             glm::vec3(-hx, hy, hz),
             glm::vec3(-hx, hy, -hz),
         }},
        // Right (+X)
        {glm::vec3(1, 0, 0),
         {
             glm::vec3(hx, -hy, hz),
             glm::vec3(hx, -hy, -hz),
             glm::vec3(hx, hy, -hz),
             glm::vec3(hx, hy, hz),
         }},
        // Top (+Y)
        {glm::vec3(0, 1, 0),
         {
             glm::vec3(-hx, hy, hz),
             glm::vec3(hx, hy, hz),
             glm::vec3(hx, hy, -hz),
             glm::vec3(-hx, hy, -hz),
         }},
        // Bottom (-Y)
        {glm::vec3(0, -1, 0),
         {
             glm::vec3(-hx, -hy, -hz),
             glm::vec3(hx, -hy, -hz),
             glm::vec3(hx, -hy, hz),
             glm::vec3(-hx, -hy, hz),
         }},
    };

    for (const auto &face : faces) {
        unsigned int start_index = static_cast<unsigned int>(vertices.size());
        for (int i = 0; i < 4; ++i) {
            vertices.push_back(face.corners[i]);
            normals.push_back(face.normal);
        }
        indices.insert(indices.end(),
                       {start_index, start_index + 1, start_index + 2, start_index, start_index + 2, start_index + 3});
    }

    return draw_info::IVPNormals(std::move(indices), std::move(vertices), std::move(normals));
}

draw_info::IVPNormals generate_cone(int segments, float height, float radius) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;

    float half_height = height / 2.0f;
    float angle_increment = 2.0f * std::numbers::pi / segments;

    // Apex vertex (index 0)
    glm::vec3 apex = glm::vec3(0.0f, half_height, 0.0f);
    vertices.push_back(apex);
    normals.push_back(glm::normalize(glm::vec3(0.0f, radius, 0.0f))); // Upward, slightly smoothed

    // Bottom center vertex (index 1)
    glm::vec3 bottom_center = glm::vec3(0.0f, -half_height, 0.0f);
    vertices.push_back(bottom_center);
    normals.push_back(glm::vec3(0.0f, -1.0f, 0.0f)); // Pointing down

    // Bottom ring vertices (indices 2..)
    std::vector<glm::vec3> base_ring;
    for (int i = 0; i < segments; ++i) {
        float angle = i * angle_increment;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        glm::vec3 pos = glm::vec3(x, -half_height, z);
        vertices.push_back(pos);
        base_ring.push_back(pos);

        // Bottom face normal (flat)
        normals.push_back(glm::vec3(0.0f, -1.0f, 0.0f));
    }

    // Side face normals for each base ring vertex
    for (int i = 0; i < segments; ++i) {
        glm::vec3 p = base_ring[i];
        glm::vec3 dir = glm::normalize(glm::vec3(p.x, 0.0f, p.z));
        glm::vec3 sloped_normal = glm::normalize(glm::vec3(dir.x, radius / height, dir.z)); // Estimate
        normals[2 + i] = sloped_normal;
    }

    // Bottom face indices (triangle fan)
    for (int i = 0; i < segments; ++i) {
        indices.push_back(1); // Bottom center
        indices.push_back(2 + i);
        indices.push_back(2 + ((i + 1) % segments));
    }

    // Side face indices (triangle fan from apex)
    for (int i = 0; i < segments; ++i) {
        indices.push_back(0); // Apex
        indices.push_back(2 + ((i + 1) % segments));
        indices.push_back(2 + i);
    }

    return draw_info::IVPNormals(std::move(indices), std::move(vertices), std::move(normals));
}

draw_info::IndexedVertexPositions generate_cylinder_between(const glm::vec3 &p1, const glm::vec3 &p2, int segments,
                                                            float radius) {
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    glm::vec3 axis = p2 - p1;
    float height = glm::length(axis);
    glm::vec3 up = glm::normalize(axis);

    // create an arbitrary perpendicular vector
    glm::vec3 ortho = glm::cross(up, glm::vec3(1.0f, 0.0f, 0.0f));
    if (glm::length(ortho) < 1e-6f) {
        ortho = glm::cross(up, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    ortho = glm::normalize(ortho);
    glm::vec3 tangent = glm::cross(up, ortho);

    float angle_increment = 2.0f * std::numbers::pi / segments;

    // top and bottom center vertices
    vertices.push_back(p2); // top center
    vertices.push_back(p1); // bottom center

    // generate vertices for top and bottom circles
    for (int i = 0; i < segments; ++i) {
        float angle = i * angle_increment;
        glm::vec3 offset = radius * (cos(angle) * ortho + sin(angle) * tangent);

        vertices.push_back(p2 + offset); // top circle vertex
        vertices.push_back(p1 + offset); // bottom circle vertex
    }

    // generate indices for the top face
    for (int i = 0; i < segments; ++i) {
        indices.push_back(0);
        indices.push_back(2 + i * 2);
        indices.push_back(2 + ((i * 2 + 2) % (segments * 2)));
    }

    // generate indices for the bottom face
    for (int i = 0; i < segments; ++i) {
        indices.push_back(1);
        indices.push_back(3 + ((i * 2) % (segments * 2)));
        indices.push_back(3 + ((i * 2 + 2) % (segments * 2)));
    }

    // generate indices for the side faces
    for (int i = 0; i < segments; ++i) {
        int top1 = 2 + (i * 2);
        int bottom1 = 3 + (i * 2);
        int top2 = 2 + ((i * 2 + 2) % (segments * 2));
        int bottom2 = 3 + ((i * 2 + 2) % (segments * 2));

        indices.push_back(top1);
        indices.push_back(bottom1);
        indices.push_back(top2);

        indices.push_back(top2);
        indices.push_back(bottom1);
        indices.push_back(bottom2);
    }

    return {indices, vertices};
}

draw_info::IVPNormals generate_cylinder(int segments, float height, float radius) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;
    float half_height = height / 2.0f;
    float angle_increment = 2.0f * std::numbers::pi / segments;

    // Top center vertex
    vertices.push_back(glm::vec3(0.0f, half_height, 0.0f));
    normals.push_back(glm::vec3(0.0f, 1.0f, 0.0f)); // Up normal

    // Bottom center vertex
    vertices.push_back(glm::vec3(0.0f, -half_height, 0.0f));
    normals.push_back(glm::vec3(0.0f, -1.0f, 0.0f)); // Down normal

    // Generate vertices for top and bottom circles
    for (int i = 0; i < segments; ++i) {
        float angle = i * angle_increment;
        float x = radius * cos(angle);
        float z = radius * sin(angle);

        // Top circle
        vertices.push_back(glm::vec3(x, half_height, z));
        normals.push_back(glm::vec3(0.0f, 1.0f, 0.0f)); // Up normal for top face

        // Bottom circle
        vertices.push_back(glm::vec3(x, -half_height, z));
        normals.push_back(glm::vec3(0.0f, -1.0f, 0.0f)); // Down normal for bottom face
    }

    // Add side vertices (duplicate positions but with radial normals)
    for (int i = 0; i < segments; ++i) {
        float angle = i * angle_increment;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        glm::vec3 radial_normal = glm::normalize(glm::vec3(x, 0.0f, z));

        // Top side vertex
        vertices.push_back(glm::vec3(x, half_height, z));
        normals.push_back(radial_normal);

        // Bottom side vertex
        vertices.push_back(glm::vec3(x, -half_height, z));
        normals.push_back(radial_normal);
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

    // Generate indices for the side faces using the side vertices
    int side_vertex_offset = 2 + segments * 2; // Offset to side vertices
    for (int i = 0; i < segments; ++i) {
        int top1 = side_vertex_offset + (i * 2);
        int bottom1 = side_vertex_offset + (i * 2) + 1;
        int top2 = side_vertex_offset + ((i + 1) % segments) * 2;
        int bottom2 = side_vertex_offset + ((i + 1) % segments) * 2 + 1;

        // First triangle
        indices.push_back(top1);
        indices.push_back(bottom1);
        indices.push_back(top2);

        // Second triangle
        indices.push_back(top2);
        indices.push_back(bottom1);
        indices.push_back(bottom2);
    }

    return draw_info::IVPNormals(indices, vertices, normals);
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
// Function to compute normals for sphere vertices
std::vector<glm::vec3> compute_sphere_normals(const std::vector<glm::vec3> &vertices) {
    std::vector<glm::vec3> normals;
    normals.reserve(vertices.size());

    for (const auto &vertex : vertices) {
        // For a sphere centered at origin, the normal is just the normalized position vector
        normals.push_back(glm::normalize(vertex));
    }

    return normals;
}

draw_info::IVPNormals generate_icosphere(int subdivisions, float radius) {
    std::vector<glm::vec3> vertices = generate_initial_icosahedron_vertices(radius);
    std::vector<unsigned int> indices = generate_initial_icosahedron_indices();

    subdivide_icosahedron(subdivisions, vertices, indices, radius);

    auto normals = compute_sphere_normals(vertices);

    return {indices, vertices, normals};
}

// Simple noise function for terrain generation
float noise(float x, float z, float seed = 0.0f) {
    // Simple hash-based noise function
    float n = std::sin(x * 12.9898f + z * 78.233f + seed) * 43758.5453f;
    return n - std::floor(n); // Return fractional part [0, 1]
}

// Multi-octave noise for more natural terrain
float fractal_noise(float x, float z, int octaves, float persistence, float scale, float seed = 0.0f) {
    float total = 0.0f;
    float frequency = 1.0f / scale;
    float amplitude = 1.0f;
    float max_value = 0.0f;

    for (int i = 0; i < octaves; ++i) {
        total += noise(x * frequency, z * frequency, seed + i * 100.0f) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0f;
    }

    return total / max_value; // Normalize to [0, 1]
}

draw_info::IVPNormals generate_terrain(float size_x, float size_z, int resolution_x, int resolution_z, float max_height,
                                       float base_height, int octaves, float persistence, float scale, float seed) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;

    float half_x = size_x / 2.0f;
    float half_z = size_z / 2.0f;
    float step_x = size_x / (resolution_x - 1);
    float step_z = size_z / (resolution_z - 1);

    // Generate height map
    std::vector<std::vector<float>> height_map(resolution_x, std::vector<float>(resolution_z));

    for (int x = 0; x < resolution_x; ++x) {
        for (int z = 0; z < resolution_z; ++z) {
            float world_x = -half_x + x * step_x;
            float world_z = -half_z + z * step_z;

            // Generate height using fractal noise
            float height = fractal_noise(world_x, world_z, octaves, persistence, scale, seed);
            height_map[x][z] = base_height + height * max_height;
        }
    }

    // Generate top surface vertices
    for (int x = 0; x < resolution_x; ++x) {
        for (int z = 0; z < resolution_z; ++z) {
            float world_x = -half_x + x * step_x;
            float world_z = -half_z + z * step_z;
            float height = height_map[x][z];

            vertices.push_back(glm::vec3(world_x, height, world_z));

            // Calculate normal using neighboring heights
            glm::vec3 normal(0.0f, 1.0f, 0.0f); // Default up normal

            if (x > 0 && x < resolution_x - 1 && z > 0 && z < resolution_z - 1) {
                // Calculate gradients
                float height_left = height_map[x - 1][z];
                float height_right = height_map[x + 1][z];
                float height_down = height_map[x][z - 1];
                float height_up = height_map[x][z + 1];

                glm::vec3 tangent_x = glm::normalize(glm::vec3(2.0f * step_x, height_right - height_left, 0.0f));
                glm::vec3 tangent_z = glm::normalize(glm::vec3(0.0f, height_up - height_down, 2.0f * step_z));

                normal = glm::normalize(glm::cross(tangent_x, tangent_z));
            }

            normals.push_back(normal);
        }
    }

    // Generate bottom surface vertices (flat)
    for (int x = 0; x < resolution_x; ++x) {
        for (int z = 0; z < resolution_z; ++z) {
            float world_x = -half_x + x * step_x;
            float world_z = -half_z + z * step_z;

            vertices.push_back(glm::vec3(world_x, base_height - max_height, world_z));
            normals.push_back(glm::vec3(0.0f, -1.0f, 0.0f)); // Down normal
        }
    }

    // Generate side wall vertices
    int top_vertex_count = resolution_x * resolution_z;
    int bottom_vertex_count = resolution_x * resolution_z;

    // Front wall (Z = +half_z)
    for (int x = 0; x < resolution_x; ++x) {
        int z = resolution_z - 1;
        float world_x = -half_x + x * step_x;
        float top_height = height_map[x][z];
        float bottom_height = base_height - max_height;

        vertices.push_back(glm::vec3(world_x, top_height, half_z));
        vertices.push_back(glm::vec3(world_x, bottom_height, half_z));
        normals.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
        normals.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
    }

    // Back wall (Z = -half_z)
    for (int x = 0; x < resolution_x; ++x) {
        int z = 0;
        float world_x = -half_x + x * step_x;
        float top_height = height_map[x][z];
        float bottom_height = base_height - max_height;

        vertices.push_back(glm::vec3(world_x, top_height, -half_z));
        vertices.push_back(glm::vec3(world_x, bottom_height, -half_z));
        normals.push_back(glm::vec3(0.0f, 0.0f, -1.0f));
        normals.push_back(glm::vec3(0.0f, 0.0f, -1.0f));
    }

    // Left wall (X = -half_x)
    for (int z = 0; z < resolution_z; ++z) {
        int x = 0;
        float world_z = -half_z + z * step_z;
        float top_height = height_map[x][z];
        float bottom_height = base_height - max_height;

        vertices.push_back(glm::vec3(-half_x, top_height, world_z));
        vertices.push_back(glm::vec3(-half_x, bottom_height, world_z));
        normals.push_back(glm::vec3(-1.0f, 0.0f, 0.0f));
        normals.push_back(glm::vec3(-1.0f, 0.0f, 0.0f));
    }

    // Right wall (X = +half_x)
    for (int z = 0; z < resolution_z; ++z) {
        int x = resolution_x - 1;
        float world_z = -half_z + z * step_z;
        float top_height = height_map[x][z];
        float bottom_height = base_height - max_height;

        vertices.push_back(glm::vec3(half_x, top_height, world_z));
        vertices.push_back(glm::vec3(half_x, bottom_height, world_z));
        normals.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
        normals.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
    }

    // Generate indices for top surface
    for (int x = 0; x < resolution_x - 1; ++x) {
        for (int z = 0; z < resolution_z - 1; ++z) {
            unsigned int top_left = x * resolution_z + z;
            unsigned int top_right = (x + 1) * resolution_z + z;
            unsigned int bottom_left = x * resolution_z + (z + 1);
            unsigned int bottom_right = (x + 1) * resolution_z + (z + 1);

            // Two triangles per quad
            indices.insert(indices.end(), {top_left, bottom_left, top_right});
            indices.insert(indices.end(), {top_right, bottom_left, bottom_right});
        }
    }

    // Generate indices for bottom surface
    for (int x = 0; x < resolution_x - 1; ++x) {
        for (int z = 0; z < resolution_z - 1; ++z) {
            unsigned int base_offset = top_vertex_count;
            unsigned int top_left = base_offset + x * resolution_z + z;
            unsigned int top_right = base_offset + (x + 1) * resolution_z + z;
            unsigned int bottom_left = base_offset + x * resolution_z + (z + 1);
            unsigned int bottom_right = base_offset + (x + 1) * resolution_z + (z + 1);

            // Two triangles per quad (reversed winding for bottom)
            indices.insert(indices.end(), {top_left, top_right, bottom_left});
            indices.insert(indices.end(), {top_right, bottom_right, bottom_left});
        }
    }

    // Generate indices for side walls
    unsigned int wall_vertex_offset = top_vertex_count + bottom_vertex_count;

    // Front wall
    for (int x = 0; x < resolution_x - 1; ++x) {
        unsigned int base = wall_vertex_offset + x * 2;
        indices.insert(indices.end(), {base, base + 1, base + 2});
        indices.insert(indices.end(), {base + 2, base + 1, base + 3});
    }
    wall_vertex_offset += resolution_x * 2;

    // Back wall
    for (int x = 0; x < resolution_x - 1; ++x) {
        unsigned int base = wall_vertex_offset + x * 2;
        indices.insert(indices.end(), {base, base + 2, base + 1});
        indices.insert(indices.end(), {base + 2, base + 3, base + 1});
    }
    wall_vertex_offset += resolution_x * 2;

    // Left wall
    for (int z = 0; z < resolution_z - 1; ++z) {
        unsigned int base = wall_vertex_offset + z * 2;
        indices.insert(indices.end(), {base, base + 2, base + 1});
        indices.insert(indices.end(), {base + 2, base + 3, base + 1});
    }
    wall_vertex_offset += resolution_z * 2;

    // Right wall
    for (int z = 0; z < resolution_z - 1; ++z) {
        unsigned int base = wall_vertex_offset + z * 2;
        indices.insert(indices.end(), {base, base + 1, base + 2});
        indices.insert(indices.end(), {base + 2, base + 1, base + 3});
    }

    return draw_info::IVPNormals(std::move(indices), std::move(vertices), std::move(normals));
}

glm::vec3 f(double t) { return glm::vec3(std::cos(t), std::sin(t), t); }

glm::vec3 compute_tangent_finite_difference(std::function<glm::vec3(double)> f, double t, double delta) {
    glm::vec3 forward = f(t + delta);
    glm::vec3 backward = f(t - delta);
    return (forward - backward) / static_cast<float>(2.0f * delta); // central difference
}

std::vector<std::pair<glm::vec3, glm::vec3>> sample_points_and_tangents(std::function<glm::vec3(double)> f,
                                                                        double t_start, double t_end, double step_size,
                                                                        double finite_diff_delta) {
    std::vector<std::pair<glm::vec3, glm::vec3>> samples;

    for (double t = t_start; t <= t_end; t += step_size) {
        glm::vec3 point = f(t);
        glm::vec3 tangent = compute_tangent_finite_difference(f, t, finite_diff_delta);
        samples.push_back({point, tangent});
    }

    return samples;
}

draw_info::IndexedVertexPositions connect_points_by_rectangles(const std::vector<glm::vec2> &points) {
    if (points.size() < 2)
        return {};

    std::vector<draw_info::IndexedVertexPositions> all_rects;
    all_rects.reserve(points.size() - 1);

    for (size_t i = 0; i < points.size() - 1; i++) {
        const auto &pa = points[i];
        const auto &pb = points[i + 1];
        all_rects.push_back(generate_rectangle_between_2d(pa, pb, 0.001));
    }

    return merge_ivps(all_rects);
}

draw_info::IndexedVertexPositions generate_function_visualization(std::function<glm::vec3(double)> f, double t_start,
                                                                  double t_end, double step_size,
                                                                  double finite_diff_delta, float radius,
                                                                  int segments) {
    return generate_segmented_cylinder(sample_points_and_tangents(f, t_start, t_end, step_size, finite_diff_delta),
                                       radius, segments);
}

draw_info::IndexedVertexPositions generate_segmented_cylinder(const std::vector<std::pair<glm::vec3, glm::vec3>> &path,
                                                              float radius, int segments) {

    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    if (path.size() < 2)
        return {indices, vertices};

    unsigned int vertex_offset = 0;

    for (size_t i = 0; i < path.size() - 1; ++i) {
        glm::vec3 p0 = path[i].first;
        glm::vec3 p1 = path[i + 1].first;
        glm::vec3 tangent = glm::normalize(p1 - p0);

        float angle_increment = 2.0f * std::numbers::pi / segments;

        // Generate an arbitrary perpendicular vector using Gram-Schmidt
        glm::vec3 arbitrary = (std::abs(tangent.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 normal = glm::normalize(glm::cross(tangent, arbitrary));
        glm::vec3 binormal = glm::normalize(glm::cross(tangent, normal));

        std::vector<unsigned int> segment_indices;

        for (int j = 0; j < segments; ++j) {
            float angle = j * angle_increment;
            glm::vec3 offset = radius * (cos(angle) * normal + sin(angle) * binormal);

            vertices.push_back(p0 + offset);
            vertices.push_back(p1 + offset);

            if (j > 0) {
                // Form two triangles per segment
                indices.push_back(vertex_offset + 2 * (j - 1));
                indices.push_back(vertex_offset + 2 * j);
                indices.push_back(vertex_offset + 2 * (j - 1) + 1);

                indices.push_back(vertex_offset + 2 * (j - 1) + 1);
                indices.push_back(vertex_offset + 2 * j);
                indices.push_back(vertex_offset + 2 * j + 1);
            }
        }

        // Close the ring
        indices.push_back(vertex_offset + 2 * (segments - 1));
        indices.push_back(vertex_offset);
        indices.push_back(vertex_offset + 2 * (segments - 1) + 1);

        indices.push_back(vertex_offset + 2 * (segments - 1) + 1);
        indices.push_back(vertex_offset);
        indices.push_back(vertex_offset + 1);

        vertex_offset += 2 * segments;
    }

    return {indices, vertices};
}

// think about lines as rungs of a ladder missing its sides, this function fills it all in with quads
draw_info::IndexedVertexPositions generate_quad_strip(const std::vector<std::pair<glm::vec3, glm::vec3>> &lines) {
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    if (lines.size() < 2)
        return {indices, vertices};

    unsigned int vertex_offset = 0;

    for (size_t i = 0; i < lines.size() - 1; ++i) {
        glm::vec3 p0a = lines[i].first;
        glm::vec3 p0b = lines[i].second;
        glm::vec3 p1a = lines[i + 1].first;
        glm::vec3 p1b = lines[i + 1].second;

        // Add vertices
        vertices.push_back(p0a);
        vertices.push_back(p0b);
        vertices.push_back(p1a);
        vertices.push_back(p1b);

        // Create two triangles to form a quad
        indices.push_back(vertex_offset);
        indices.push_back(vertex_offset + 2);
        indices.push_back(vertex_offset + 1);

        indices.push_back(vertex_offset + 1);
        indices.push_back(vertex_offset + 2);
        indices.push_back(vertex_offset + 3);

        vertex_offset += 4;
    }

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

draw_info::IndexedVertexPositions merge_ivps(const std::vector<draw_info::IndexedVertexPositions> &ivps) {
    draw_info::IndexedVertexPositions merged_ivp;

    for (const auto &ivp : ivps) {
        merge_ivps(merged_ivp, ivp);
    }

    return merged_ivp;
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

// NOTE: the surface normal is not unique here, this is because we are already constrained by a, b thus
/*
 *                 SN                  SN2
 *                  \                 /
 *                   \               /
 *                    \             /                    --- B
 *                     \           /                 ---/
 *  SN1                 \         /               ---/
 *   ---\                \       /            ---/
 *       -----\           \     /          ---/
 *             -----\      \   /       ---/
 *                   -----\ \ /     ---/
 *                           . ----/
 *                         ---/--\
 *                     ---/       ---\
 *                 ---/               ---\
 *             ---/                       ---\
 *         ---/                               ---\
 *     ---/                                       --- width_dir
 *  --/
 * A
 *
 * given the direction vector B - A, then note that there are a family of surface normals yielding the same width
 * dir, note that the surface normal and B - A define a plane that passes through both lines, and any vector such as
 * SN1 or SN2 also will create width_dir with the cross product is used, this we don't require that the surface
 * normal be actually perpendicular to B - A
 *
 *
 */
std::vector<glm::vec3> generate_rectangle_vertices_from_points(const glm::vec3 &point_a, const glm::vec3 &point_b,
                                                               const glm::vec3 &surface_normal, float height) {
    glm::vec3 width_dir = glm::normalize(point_b - point_a);
    glm::vec3 height_dir = glm::normalize(glm::cross(surface_normal, width_dir));
    glm::vec3 half_height_vec = (height / 2.0f) * height_dir;

    return {
        point_a + half_height_vec, // top left
        point_a - half_height_vec, // bottom left
        point_b - half_height_vec, // bottom right
        point_b + half_height_vec  // top right
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

std::vector<glm::vec2> generate_rectangle_texture_coordinates_flipped_vertically() {
    return {
        glm::vec2(1.0f, 1.0f), // Top-right (was Bottom-right)
        glm::vec2(1.0f, 0.0f), // Bottom-right (was Top-right)
        glm::vec2(0.0f, 0.0f), // Bottom-left (was Top-left)
        glm::vec2(0.0f, 1.0f)  // Top-left (was Bottom-left)
    };
}

draw_info::IndexedVertexPositions generate_circle(const glm::vec3 center, float radius, unsigned int num_sides) {
    return generate_n_gon(center, radius, num_sides);
}

draw_info::IndexedVertexPositions generate_n_gon(const glm::vec3 center, float radius, unsigned int num_sides) {
    return {generate_n_gon_indices(num_sides), generate_n_gon_vertices(center, radius, num_sides)};
}

std::vector<unsigned int> generate_n_gon_indices(unsigned int num_sides) {
    assert(num_sides >= 3);

    std::vector<unsigned int> indices;
    for (unsigned int i = 1; i <= num_sides; ++i) {
        unsigned int next = (i % num_sides) + 1; // wrap around to 1 after num_sides
        indices.push_back(0);                    // center
        indices.push_back(i);                    // current outer vertex
        indices.push_back(next);                 // next outer vertex
    }
    return indices;
}

/**
 *
 * @brief generate n equally spaced points on the unit circle
 *
 *
 */

std::vector<glm::vec3> generate_n_gon_vertices(const glm::vec3 &center, float radius, unsigned int num_sides) {
    assert(num_sides >= 3);

    std::vector<glm::vec3> vertices;
    vertices.push_back(center); // center point for triangle fan

    float angle_increment = 2.0f * glm::pi<float>() / static_cast<float>(num_sides);

    for (unsigned int i = 0; i < num_sides; ++i) {
        float angle = i * angle_increment;
        float x = center.x + radius * std::cos(angle);
        float y = center.y + radius * std::sin(angle);
        vertices.emplace_back(x, y, center.z);
    }

    return vertices;
}

draw_info::IndexedVertexPositions generate_annulus(float center_x, float center_y, float outer_radius,
                                                   float inner_radius, int num_segments, float percent) {
    return {generate_annulus_indices(num_segments, percent),
            generate_annulus_vertices(center_x, center_y, outer_radius, inner_radius, num_segments, percent)};
}

std::vector<glm::vec3> generate_annulus_vertices(float center_x, float center_y, float outer_radius, float inner_radius,
                                                 int num_segments, float percent) {
    std::vector<glm::vec3> vertices;
    float full_circle = 2.0f * std::numbers::pi;
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

draw_info::IndexedVertexPositions generate_star(float center_x, float center_y, float outer_radius, float inner_radius,
                                                int num_star_tips, bool blunt_tips) {
    return {
        generate_star_indices(num_star_tips, blunt_tips),
        generate_star_vertices(center_x, center_y, outer_radius, inner_radius, num_star_tips, blunt_tips),
    };
}

std::vector<glm::vec3> generate_star_vertices(float center_x, float center_y, float outer_radius, float inner_radius,
                                              int num_star_tips, bool blunt_tips) {

    int star_multiplier = (blunt_tips ? 3 : 2);
    int num_vertices_required = star_multiplier * num_star_tips;

    int inner_radius_offset = 1;

    float full_rotation = 2 * std::numbers::pi;

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
    float phi = std::numbers::pi * (std::sqrt(5.0) - 1.0);

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

draw_info::IndexedVertexPositions generate_3d_arrow_with_ratio(const glm::vec3 &start, const glm::vec3 &end,
                                                               int num_segments, float length_thickness_ratio) {
    glm::vec3 direction = end - start;
    float length = glm::length(direction);

    // calculate the stem thickness (height) using the width-to-height ratio
    float stem_thickness = length * length_thickness_ratio;

    return generate_3d_arrow(start, end, num_segments, stem_thickness);
}

draw_info::IndexedVertexPositions generate_3d_arrow(const glm::vec3 &start, const glm::vec3 &end, int num_segments,
                                                    float stem_thickness) {

    glm::vec3 direction = end - start;

    float length = glm::length(direction);

    // compute the tip length as 10% of the direction vector
    float tip_length = 0.3f * length;
    float stem_length = length - tip_length;

    // generate the stem and tip using the new functions
    auto stem = vertex_geometry::generate_cylinder_between(
        start, start + direction - glm::normalize(direction) * tip_length, num_segments, stem_thickness);

    auto tip = vertex_geometry::generate_cone_between(start + direction - glm::normalize(direction) * tip_length, end,
                                                      num_segments, 2 * stem_thickness);

    vertex_geometry::merge_ivps(stem, tip);

    return stem;
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

void scale_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &scale_vector, const glm::vec3 &origin) {
    for (auto &vertex : vertices) {
        vertex = origin + (vertex - origin) * scale_vector;
    }
}

std::vector<glm::vec3> scale_vertices(const std::vector<glm::vec3> &vertices, const glm::vec3 &scale_vector,
                                      const glm::vec3 &origin) {
    std::vector<glm::vec3> scaled_vertices = vertices;
    for (auto &vertex : scaled_vertices) {
        vertex = origin + (vertex - origin) * scale_vector;
    }
    return scaled_vertices;
}

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

glm::mat4 yaw_pitch_roll(float yaw, float pitch, float roll) { // rads
    float cy = cos(yaw), sy = sin(yaw);
    float cp = cos(pitch), sp = sin(pitch);
    float cr = cos(roll), sr = sin(roll);

    return glm::mat4(cy * cr + sy * sp * sr, sr * cp, sy * cr - cy * sp * sr, 0.0f, -cy * sr + sy * sp * cr, cr * cp,
                     -sy * sr - cy * sp * cr, 0.0f, sy * cp, -sp, cy * cp, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

void rotate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &rotation_turns) {
    glm::vec3 rotation_radians = rotation_turns * glm::two_pi<float>(); // Convert turns to radians
    glm::mat4 rotation_matrix = yaw_pitch_roll(rotation_radians.y, rotation_radians.x, rotation_radians.z);

    for (auto &vertex : vertices) {
        vertex = glm::vec3(rotation_matrix * glm::vec4(vertex, 1.0f));
    }
}

void increment_indices_in_place(std::vector<unsigned int> &indices, unsigned int increase) {
    for (auto &index : indices) {
        index += increase;
    }
}
} // namespace vertex_geometry
