#ifndef VERTEX_GEOMETRY_HPP
#define VERTEX_GEOMETRY_HPP

#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

struct IndexedVertices {
    std::vector<glm::vec3> vertices;   // Vertices of the grid
    std::vector<unsigned int> indices; // Flattened indices of the grid
};

class Rectangle {
  public:
    glm::vec3 center; // Center position
    float width;      // Width of the rectangle
    float height;     // Height of the rectangle
    IndexedVertices get_ivs();
};

class Grid {
  public:
    Grid(int rows, int cols, float width = 2.0f, float height = 2.0f, float origin_x = 0.0f, float origin_y = 0.0f);
    Grid(int rows, int cols, const Rectangle &rect);

    Rectangle get_at(int col, int row) const;

    /**
     * @brief Get all rectangles inside a bounding box specified by two row-column pairs.
     *
     * @param row1 Starting row index (0-based).
     * @param col1 Starting column index (0-based).
     * @param row2 Ending row index (0-based).
     * @param col2 Ending column index (0-based).
     * @return A vector containing all rectangles within the specified bounding box.
     * @throws std::out_of_range if any row or column indices are out of bounds.
     */
    std::vector<Rectangle> get_rectangles_in_bounding_box(int row1, int col1, int row2, int col2) const;

    /**
     * @brief Get all rectangles in a specific row.
     * @param row Row index (0-based).
     * @return A vector of rectangles in the specified row.
     * @throws std::out_of_range if the row is out of bounds.
     */
    std::vector<Rectangle> get_row(int row) const;

    /**
     * @brief Get all rectangles in a specific column.
     * @param col Column index (0-based).
     * @return A vector of rectangles in the specified column.
     * @throws std::out_of_range if the column is out of bounds.
     */
    std::vector<Rectangle> get_column(int col) const;

    const int rows; // Number of rows in the grid
    const int cols; // Number of columns in the grid

  private:
    float grid_width;  // Total width of the grid in NDC
    float grid_height; // Total height of the grid in NDC
    float origin_x;    // X-coordinate of the grid's origin
    float origin_y;    // Y-coordinate of the grid's origin
    float rect_width;  // Width of each rectangle
    float rect_height; // Height of each rectangle
};

Rectangle create_rectangle_from_corners(const glm::vec3 top_left, const glm::vec3 top_right,
                                        const glm::vec3 bottom_left, const glm::vec3 bottom_right);
Rectangle create_rectangle(float x_pos, float y_pos, float width, float height);
Rectangle create_rectangle_from_top_left(const glm::vec3 &top_left, float width, float height);
Rectangle create_rectangle_from_top_right(const glm::vec3 &top_right, float width, float height);
Rectangle create_rectangle_from_bottom_left(const glm::vec3 &bottom_left, float width, float height);
Rectangle create_rectangle_from_bottom_right(const glm::vec3 &bottom_right, float width, float height);
Rectangle create_rectangle_from_center(const glm::vec3 &center, float width, float height);

std::vector<glm::vec3> generate_rectangle_normals();
std::vector<Rectangle> generate_grid_rectangles(const glm::vec3 &center_position, float base_width, float base_height,
                                                int num_rectangles_x, int num_rectangles_y, float spacing);

IndexedVertices generate_grid(const glm::vec3 &center_position, float base_width, float base_height,
                              int num_rectangles_x, int num_rectangles_y, float spacing);

IndexedVertices generate_grid(const glm::vec3 &center_position, float width, float height, int num_rectangles_x,
                              int num_rectangles_y, float spacing);

std::vector<unsigned int> flatten_and_increment_indices(const std::vector<std::vector<unsigned int>> &indices);

std::vector<glm::vec3> generate_square_vertices(float center_x, float center_y, float side_length);
std::vector<unsigned int> generate_square_indices();

std::vector<glm::vec3> generate_rectangle_vertices(float center_x, float center_y, float width, float height);
std::vector<unsigned int> generate_rectangle_indices();
std::vector<glm::vec2> generate_rectangle_texture_coordinates();

std::vector<glm::vec3> generate_rectangle_vertices_3d(const glm::vec3 &center, const glm::vec3 &width_dir,
                                                      const glm::vec3 &height_dir, float width, float height);

std::vector<glm::vec3> generate_arrow_vertices(glm::vec2 start, glm::vec2 end, float stem_thickness, float tip_length);
std::vector<unsigned int> generate_arrow_indices();

void scale_vertices_in_place(std::vector<glm::vec3> &vertices, float scale_factor);

std::vector<glm::vec3> generate_n_gon_flattened_vertices(int n);

int get_num_flattened_vertices_in_n_gon(int n);

std::vector<glm::vec3> generate_fibonacci_sphere_vertices(int num_samples, float scale);

void translate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &translation);
void increment_indices_in_place(std::vector<unsigned int> &indices, unsigned int increase);

#endif // VERTEX_GEOMETRY_HPP
