#ifndef VERTEX_GEOMETRY_HPP
#define VERTEX_GEOMETRY_HPP

#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "sbpt_generated_includes.hpp"

// NOTE: One day I want to move grid font into this, but here all we want to do is just store more hpp/cpp files in this
// one rather than forcing all that into this file as well.

namespace vertex_geometry {

enum class TriangulationMode {
    CentralFan, // Triangulate from center point
    VertexFan   // Triangulate from an existing vertex
};

class NGon {
  public:
    // Construct from explicit points
    NGon(const std::vector<glm::vec3> &pts, TriangulationMode mode = TriangulationMode::CentralFan)
        : points(pts), triangulation_mode(mode) {
        if (points.size() < 3)
            throw std::runtime_error("Ngon must have at least 3 vertices");
    }

    // Construct a regular ngon given radius, plane normal, and center offset
    NGon(std::size_t n_vertices, float radius = 1.0f, const glm::vec3 &normal = glm::vec3(0, 0, 1),
         const glm::vec3 &offset = glm::vec3(0, 0, 0), TriangulationMode mode = TriangulationMode::CentralFan)
        : triangulation_mode(mode) {
        if (n_vertices < 3)
            throw std::runtime_error("Ngon must have at least 3 vertices");

        points.resize(n_vertices);
        glm::vec3 u, v;
        make_basis_from_normal(normal, u, v);

        float angle_step = glm::two_pi<float>() / static_cast<float>(n_vertices);
        for (std::size_t i = 0; i < n_vertices; ++i) {
            float angle = i * angle_step;
            points[i] = offset + radius * (std::cos(angle) * u + std::sin(angle) * v);
        }
    }

    const glm::vec3 &operator[](std::size_t index) const { return points[index]; }
    glm::vec3 &operator[](std::size_t index) { return points[index]; }

    std::size_t size() const { return points.size(); }
    const std::vector<glm::vec3> &get_points() const { return points; }

    TriangulationMode get_triangulation_mode() const { return triangulation_mode; }

  private:
    std::vector<glm::vec3> points;
    TriangulationMode triangulation_mode;

    static void make_basis_from_normal(const glm::vec3 &normal, glm::vec3 &u, glm::vec3 &v) {
        glm::vec3 n = glm::normalize(normal);
        glm::vec3 temp = (std::abs(n.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        u = glm::normalize(glm::cross(n, temp));
        v = glm::normalize(glm::cross(n, u));
    }
};

class AxisAlignedBoundingBox {
  public:
    AxisAlignedBoundingBox(const std::vector<glm::vec3> &xyz_positions) {
        min = glm::vec3(std::numeric_limits<float>::max());
        max = glm::vec3(std::numeric_limits<float>::lowest());

        for (const auto &v : xyz_positions) {
            min = glm::min(min, v);
            max = glm::max(max, v);
        }
    }

    glm::vec3 min;
    glm::vec3 max;

    std::array<glm::vec3, 8> get_corners() const {
        return {
            glm::vec3(min.x, min.y, min.z), glm::vec3(max.x, min.y, min.z), glm::vec3(min.x, max.y, min.z),
            glm::vec3(max.x, max.y, min.z), glm::vec3(min.x, min.y, max.z), glm::vec3(max.x, min.y, max.z),
            glm::vec3(min.x, max.y, max.z), glm::vec3(max.x, max.y, max.z),
        };
    }

    draw_info::IndexedVertexPositions get_ivp();
};

draw_info::IndexedVertexPositions triangulate_ngon(const NGon &ngon);
draw_info::IndexedVertexPositions connect_ngons(const NGon &a, const NGon &b);

// Triangulate any Ngon dynamically

class Rectangle {
  public:
    // by default we use width height 2 to take up the full [-1, 1] x [-1, 1] ndc space
    Rectangle(glm::vec3 center = glm::vec3(0), float width = 2, float height = 2)
        : center(center), width(width), height(height) {}
    glm::vec3 center; // Center position
    float width;      // Width of the rectangle
    float height;     // Height of the rectangle
    draw_info::IndexedVertexPositions get_ivs() const;
    friend std::ostream &operator<<(std::ostream &os, const Rectangle &rect);

    glm::vec3 get_top_left() const;
    glm::vec3 get_top_center() const;
    glm::vec3 get_top_right() const;
    glm::vec3 get_center_left() const;
    glm::vec3 get_center_right() const;
    glm::vec3 get_bottom_left() const;
    glm::vec3 get_bottom_center() const;
    glm::vec3 get_bottom_right() const;
};

Rectangle expand_rectangle(const Rectangle &rect, float x_expand, float y_expand);
Rectangle shrink_rectangle(const Rectangle &rect, float x_shrink, float y_shrink);
// this function scales a rectangle keeping the left side in place
Rectangle scale_rectangle_from_left_side(const Rectangle &rect, float x_shrink, float y_shrink = 1);
Rectangle slide_rectangle(const Rectangle &rect, int x_offset, int y_offset);
Rectangle get_bounding_rectangle(const std::vector<Rectangle> &rectangles);

class Grid {
  public:
    Grid(int rows, int cols, float width = 2.0f, float height = 2.0f, float origin_x = 0.0f, float origin_y = 0.0f,
         float origin_z = 0.0f);
    Grid(int rows, int cols, const Rectangle &rect);

    // this is like x, y
    Rectangle get_at(int col, int row) const;
    std::vector<Rectangle> get_selection(float x0, float y0, float x1, float y1) const;
    std::vector<Rectangle> get_rectangles_in_bounding_box(int row1, int col1, int row2, int col2) const;
    std::vector<Rectangle> get_row(int row) const;
    std::vector<Rectangle> get_column(int col) const;

    const int rows; // Number of rows in the grid
    const int cols; // Number of columns in the grid

  private:
    float grid_width;  // Total width of the grid in NDC
    float grid_height; // Total height of the grid in NDC
    float origin_x;    // X-coordinate of the grid's origin
    float origin_y;    // Y-coordinate of the grid's origin
    float origin_z;    // Y-coordinate of the grid's origin
    float rect_width;  // Width of each rectangle
    float rect_height; // Height of each rectangle
};

bool circle_intersects_rect(float cx, float cy, float radius, const Rectangle &rect);
std::vector<Rectangle> get_rects_intersecting_circle(const Grid &grid, float cx, float cy, float radius);

draw_info::IndexedVertexPositions text_grid_to_rect_grid(const std::string &text_grid,
                                                         const vertex_geometry::Rectangle bounding_rect);

draw_info::IndexedVertexPositions generate_rectangle_between_2d(const glm::vec2 &p1, const glm::vec2 &p2,
                                                                float thickness);

Rectangle create_rectangle_from_corners(const glm::vec3 top_left, const glm::vec3 top_right,
                                        const glm::vec3 bottom_left, const glm::vec3 bottom_right);
Rectangle create_rectangle(float x_pos, float y_pos, float width, float height);
Rectangle create_rectangle_from_top_left(const glm::vec3 &top_left, float width, float height);
Rectangle create_rectangle_from_top_right(const glm::vec3 &top_right, float width, float height);
Rectangle create_rectangle_from_bottom_left(const glm::vec3 &bottom_left, float width, float height);
Rectangle create_rectangle_from_bottom_right(const glm::vec3 &bottom_right, float width, float height);
Rectangle create_rectangle_from_center_left(const glm::vec3 &center_left, float width, float height);

Rectangle create_rectangle_from_center(const glm::vec3 &center, float width, float height);

std::vector<Rectangle> subdivide_rectangle(const Rectangle &rect, unsigned int num_subdivisions, bool vertical = true);

// when you subdivide vertically think of it as cutting up and down like lines in a book
std::vector<Rectangle> vertical_weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights);

// when you subdivide horizontally you're cutting a carrot on a cutting board
std::vector<Rectangle> horizontal_weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights);

std::vector<Rectangle> weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights,
                                            bool vertical = true);

std::vector<glm::vec3> generate_rectangle_normals();
std::vector<Rectangle> generate_grid_rectangles(const glm::vec3 &center_position, float base_width, float base_height,
                                                int num_rectangles_x, int num_rectangles_y, float spacing);

draw_info::IndexedVertexPositions generate_grid(const glm::vec3 &center_position, float base_width, float base_height,
                                                int num_rectangles_x, int num_rectangles_y, float spacing);

draw_info::IndexedVertexPositions generate_grid(const glm::vec3 &center_position, float width, float height,
                                                int num_rectangles_x, int num_rectangles_y, float spacing);

std::vector<unsigned int> flatten_and_increment_indices(const std::vector<std::vector<unsigned int>> &indices);

draw_info::IVPNormals generate_torus(int major_segments = 64,   // Around the main ring
                                     int minor_segments = 32,   // Around the tube
                                     float major_radius = 1.0f, // Distance from center to tube center
                                     float minor_radius = 0.3f  // Radius of the tube
);

draw_info::IVPNormals generate_cube(float size = 1.0f);

draw_info::IVPNormals generate_box(float size_x = 1.0f, float size_y = 1.0f, float size_z = 1.0f);

draw_info::IVPNormals generate_cone(int segments = 8, float height = 1.0f, float radius = 0.5f);

draw_info::IndexedVertexPositions generate_cone_between(const glm::vec3 &base, const glm::vec3 &tip, int segments = 8,
                                                        float radius = 0.5);

draw_info::IVPNormals generate_cylinder(int segments = 8, float height = 1.0f, float radius = 0.5f);

draw_info::IndexedVertexPositions generate_cylinder_between(const glm::vec3 &p1, const glm::vec3 &p2, int segments,
                                                            float radius);

draw_info::IVPNormals generate_icosphere(int subdivisions, float radius);

draw_info::IVPNormals generate_terrain(float size_x = 100.0f, float size_z = 100.0f, int resolution_x = 50,
                                       int resolution_z = 50, float max_height = 5.0f, float base_height = 0.0f,
                                       int octaves = 4, float persistence = 0.5f, float scale = 50.0f,
                                       float seed = 0.0f);

draw_info::IndexedVertexPositions generate_function_visualization(std::function<glm::vec3(double)> f, double t_start,
                                                                  double t_end, double step_size,
                                                                  double finite_diff_delta, float radius = .25,
                                                                  int segments = 8);

draw_info::IndexedVertexPositions generate_segmented_cylinder(const std::vector<std::pair<glm::vec3, glm::vec3>> &path,
                                                              float radius, int segments);

draw_info::IndexedVertexPositions generate_quad_strip(const std::vector<std::pair<glm::vec3, glm::vec3>> &lines);

void merge_ivps(draw_info::IndexedVertexPositions &base_ivp, const draw_info::IndexedVertexPositions &extend_ivp);
void merge_ivps(draw_info::IndexedVertexPositions &base_ivp,
                const std::vector<draw_info::IndexedVertexPositions> &extend_ivps);
draw_info::IndexedVertexPositions merge_ivps(const std::vector<draw_info::IndexedVertexPositions> &ivps);

draw_info::IndexedVertexPositions generate_unit_cube();
std::vector<glm::vec3> generate_unit_cube_vertices();
std::vector<unsigned int> generate_cube_indices();

std::vector<glm::vec3> generate_square_vertices(float center_x, float center_y, float side_length);
std::vector<unsigned int> generate_square_indices();

draw_info::IndexedVertexPositions generate_rectangle(float center_x, float center_y, float width, float height);
std::vector<glm::vec3> generate_rectangle_vertices(float center_x, float center_y, float width, float height);
// TODO: the below shouldn't exist, instead the above should just take in z, but i don't want to bust the api right
// now
draw_info::IndexedVertexPositions generate_rectangle(float center_x, float center_y, float center_z, float width,
                                                     float height);
std::vector<glm::vec3> generate_rectangle_vertices_with_z(float center_x, float center_y, float center_z, float width,
                                                          float height);
std::vector<unsigned int> generate_rectangle_indices();
std::vector<glm::vec2> generate_rectangle_texture_coordinates();
std::vector<glm::vec2> generate_rectangle_texture_coordinates_flipped_vertically();

std::vector<glm::vec3> generate_rectangle_vertices_3d(const glm::vec3 &center, const glm::vec3 &width_dir,
                                                      const glm::vec3 &height_dir, float width, float height);
std::vector<glm::vec3> generate_rectangle_vertices_from_points(const glm::vec3 &point_a, const glm::vec3 &point_b,
                                                               const glm::vec3 &surface_normal, float height = 1);

draw_info::IndexedVertexPositions generate_3d_arrow_with_ratio(const glm::vec3 &start, const glm::vec3 &end,
                                                               int num_segments = 16,
                                                               float length_thickness_ratio = 0.07);

draw_info::IndexedVertexPositions generate_3d_arrow(const glm::vec3 &start, const glm::vec3 &end, int num_segments = 16,
                                                    float stem_thickness = 0.12);

std::vector<glm::vec3> generate_arrow_vertices(glm::vec2 start, glm::vec2 end, float stem_thickness = 0.05,
                                               float tip_length = .05);
std::vector<unsigned int> generate_arrow_indices();

std::vector<glm::vec3> scale_vertices(const std::vector<glm::vec3> &vertices, const glm::vec3 &scale_vector,
                                      const glm::vec3 &origin = glm::vec3(0.0f));
void scale_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &scale_vector,
                             const glm::vec3 &origin = glm::vec3(0.0f));
void scale_vertices_in_place(std::vector<glm::vec3> &vertices, float scale_factor);
void rotate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &rotation_turns);

// note: num sides >= 3
draw_info::IndexedVertexPositions generate_circle(const glm::vec3 center = glm::vec3(0), float radius = 1,
                                                  unsigned int num_sides = 30);
draw_info::IndexedVertexPositions generate_n_gon(const glm::vec3 center = glm::vec3(0), float radius = 1,
                                                 unsigned int num_sides = 30);
std::vector<unsigned int> generate_n_gon_indices(unsigned int num_sides);
std::vector<glm::vec3> generate_n_gon_vertices(const glm::vec3 &center, float radius = 1, unsigned int num_sides = 30);

draw_info::IndexedVertexPositions generate_annulus(float center_x, float center_y, float outer_radius,
                                                   float inner_radius, int num_segments, float percent = 1);
std::vector<glm::vec3> generate_annulus_vertices(float center_x, float center_y, float outer_radius, float inner_radius,
                                                 int num_segments, float percent = 1);
std::vector<unsigned int> generate_annulus_indices(int num_segments, float percent = 1);

// points are the points of the star
std::vector<glm::vec3> generate_star_vertices(float center_x, float center_y, float outer_radius, float inner_radius,
                                              int num_star_tips, bool blunt_tips = false);
std::vector<unsigned int> generate_star_indices(int num_star_tips, bool blunt_tips);

std::vector<glm::vec3> generate_fibonacci_sphere_vertices(int num_samples, float scale);

void translate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &translation);
void increment_indices_in_place(std::vector<unsigned int> &indices, unsigned int increase);
} // namespace vertex_geometry

#endif // VERTEX_GEOMETRY_HPP
