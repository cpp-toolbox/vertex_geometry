#ifndef VERTEX_GEOMETRY_HPP
#define VERTEX_GEOMETRY_HPP

#include <glm/glm.hpp>
#include <vector>

struct IndexedVertices {
    std::vector<glm::vec3> vertices;   // Vertices of the grid
    std::vector<unsigned int> indices; // Flattened indices of the grid
};

struct Rectangle {
    glm::vec3 center; // Center position
    float width;      // Width of the rectangle
    float height;     // Height of the rectangle
};

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

std::vector<glm::vec3> generate_arrow_vertices(glm::vec2 start, glm::vec2 end, float stem_thickness, float tip_length);
std::vector<unsigned int> generate_arrow_indices();

void scale_vertices_in_place(std::vector<glm::vec3> &vertices, float scale_factor);

std::vector<glm::vec3> generate_n_gon_flattened_vertices(int n);

int get_num_flattened_vertices_in_n_gon(int n);

std::vector<glm::vec3> generate_fibonacci_sphere_vertices(int num_samples, float scale);

void translate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &translation);
void increment_indices_in_place(std::vector<unsigned int> &indices, unsigned int increase);

#endif // VERTEX_GEOMETRY_HPP
