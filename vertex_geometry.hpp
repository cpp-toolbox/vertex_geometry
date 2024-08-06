#ifndef VERTEX_GEOMETRY_HPP
#define VERTEX_GEOMETRY_HPP

#include "glm/glm.hpp"
#include <vector>

// unsigned int num_indices_required_to_specify_retangle = 6;

std::vector<glm::vec3> generate_square_vertices(float center_x, float center_y, float side_length);

std::vector<unsigned int> generate_square_indices();

std::vector<glm::vec3> generate_rectangle_vertices(float center_x, float center_y, float width, float height);

std::vector<unsigned int> generate_rectangle_indices();

std::vector<glm::vec3> generate_arrow_vertices(glm::vec2 start, glm::vec2 end, float stem_thickness, float tip_length);

std::vector<unsigned int> generate_arrow_indices();

void scale_vertices_in_place(std::vector<glm::vec3> &vertices, float scale_factor);

std::vector<glm::vec3> generate_n_gon_flattened_vertices(int n);

int get_num_flattened_vertices_in_n_gon(int n);

void translate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &translation);

void increment_indices_in_place(std::vector<unsigned int> &indices, unsigned int increase);

#endif // VERTEX_GEOMETRY_HPP