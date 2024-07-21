#ifndef VERTEX_GEOMETRY_HPP
#define VERTEX_GEOMETRY_HPP

#include "glm/vec2.hpp"

void store_square_vertices(float *out_square, float center_x, float center_y, float side_length, int iteration);

void store_square_indices(unsigned int *out_indices, unsigned int iteration);

void
store_rectangle_vertices(float *out_square, float center_x, float center_y, float width, float height, int iteration);

void store_rectangle_indices(unsigned int *out_indices, unsigned int iteration);

void store_arrow_vertices(glm::vec2 start, glm::vec2 end, float stem_thickness, float tip_length,
                          float *out_flattened_vertices);

void store_arrow_indices(unsigned int *out_indices);

void scale_vertices(float *vertices_to_be_scaled, int num_vertices, float scale_factor);

void store_n_gon_flattened_vertices(float *out_vertices, int n);

int get_num_flattened_vertices_in_n_gon(int n);

void translate_vertices(float *flattened_vertices_to_be_scaled, int num_vertices, float x_translate, float y_translate);

#endif //VERTEX_GEOMETRY_HPP
