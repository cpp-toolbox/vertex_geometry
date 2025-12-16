#ifndef VERTEX_GEOMETRY_HPP
#define VERTEX_GEOMETRY_HPP

#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <array>

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

/**
 * @brief Represents an axis-aligned bounding box (AABB) in 3D space.
 *
 * The AABB is defined by its minimum and maximum corner points.
 * It provides utility methods to retrieve corner positions,
 * projected positions (e.g., max X/Y with z=0), and a conversion to
 * drawable vertex data.
 */
class AxisAlignedBoundingBox {
  public:
    /**
     * @brief Default constructor. Creates an uninitialized bounding box.
     */
    AxisAlignedBoundingBox() {}

    /**
     * @brief Constructs an axis-aligned bounding box from a set of 3D points.
     *
     * Iterates over all given positions to compute the minimum and maximum
     * extents of the box along each axis.
     *
     * @note if you already know min and max then just pass them in as a vector of vec3s there is no specific
     * constructor for doing that
     *
     * @param xyz_positions A list of 3D positions used to compute the bounding box.
     */
    AxisAlignedBoundingBox(const std::vector<glm::vec3> &xyz_positions) {
        min = glm::vec3(std::numeric_limits<float>::max());
        max = glm::vec3(std::numeric_limits<float>::lowest());

        for (const auto &v : xyz_positions) {
            min = glm::min(min, v);
            max = glm::max(max, v);
        }
    }

    /** @brief The minimum corner of the bounding box. */
    glm::vec3 min;

    /** @brief The maximum corner of the bounding box. */
    glm::vec3 max;

    /**
     * @brief Returns the 8 corner points of the bounding box.
     *
     * The corners are returned in the following order:
     * (min.x, min.y, min.z), (max.x, min.y, min.z),
     * (min.x, max.y, min.z), (max.x, max.y, min.z),
     * (min.x, min.y, max.z), (max.x, min.y, max.z),
     * (min.x, max.y, max.z), (max.x, max.y, max.z)
     *
     * @return An array containing all 8 corner positions.
     */
    std::array<glm::vec3, 8> get_corners() const {
        return {glm::vec3(min.x, min.y, min.z), glm::vec3(max.x, min.y, min.z), glm::vec3(min.x, max.y, min.z),
                glm::vec3(max.x, max.y, min.z), glm::vec3(min.x, min.y, max.z), glm::vec3(max.x, min.y, max.z),
                glm::vec3(min.x, max.y, max.z), glm::vec3(max.x, max.y, max.z)};
    }

    /**
     * @brief Returns the position with the maximum X and Y values.
     * @return A 3D position with (max.x, max.y, 0.0f).
     */
    glm::vec3 get_max_xy_position() const { return glm::vec3(max.x, max.y, 0.0f); }

    /**
     * @brief Returns the position with the minimum X and Y values.
     * @return A 3D position with (min.x, min.y, 0.0f).
     */
    glm::vec3 get_min_xy_position() const { return glm::vec3(min.x, min.y, 0.0f); }

    /**
     * @brief Returns the position with the maximum X and minimum Y values.
     * @return A 3D position with (max.x, min.y, 0.0f).
     */
    glm::vec3 get_maxx_miny_position() const { return glm::vec3(max.x, min.y, 0.0f); }

    /**
     * @brief Returns the position with the minimum X and maximum Y values.
     * @return A 3D position with (min.x, max.y, 0.0f).
     */
    glm::vec3 get_minx_maxy_position() const { return glm::vec3(min.x, max.y, 0.0f); }

    /**
     * @brief Converts the bounding box into indexed vertex positions for drawing.
     * @return A draw_info::IndexedVertexPositions object representing the box geometry.
     */
    draw_info::IndexedVertexPositions get_ivp();
};

class AxisAlignedBoundingBox2D {
  public:
    AxisAlignedBoundingBox2D(const std::vector<glm::vec2> &xy_positions) {
        min = glm::vec2(std::numeric_limits<float>::max());
        max = glm::vec2(std::numeric_limits<float>::lowest());

        for (const auto &v : xy_positions) {
            min = glm::min(min, v);
            max = glm::max(max, v);
        }
    }

    glm::vec2 min;
    glm::vec2 max;

    // Returns the 4 corners of the 2D bounding box
    std::array<glm::vec2, 4> get_corners() const {
        return {glm::vec2(min.x, min.y), glm::vec2(max.x, min.y), glm::vec2(min.x, max.y), glm::vec2(max.x, max.y)};
    }

    // LATER
    // draw_info::IndexedVertexPositions get_ivp();
};

draw_info::IndexedVertexPositions triangulate_ngon(const NGon &ngon);
draw_info::IndexedVertexPositions connect_ngons(const NGon &a, const NGon &b);

class Rectangle {
  public:
    // by default we use width height 2 to take up the full [-1, 1] x [-1, 1] ndc space
    Rectangle(glm::vec3 center = glm::vec3(0), float width = 2, float height = 2)
        : center(center), width(width), height(height) {}
    glm::vec3 center; // Center position
    float width;      // Width of the rectangle
    float height;     // Height of the rectangle
    // TODO: this needs to be renamed get_ivp right?
    draw_info::IndexedVertexPositions get_ivs() const;
    friend std::ostream &operator<<(std::ostream &os, const Rectangle &rect);

    glm::vec3 get_top_left() const;
    glm::vec3 get_top_center() const;
    glm::vec3 get_top_right() const;
    glm::vec3 get_left_center() const;
    glm::vec3 get_right_center() const;
    glm::vec3 get_bottom_left() const;
    glm::vec3 get_bottom_center() const;
    glm::vec3 get_bottom_right() const;
};

Rectangle expand_rectangle(const Rectangle &rect, float x_expand, float y_expand);

/**
 * @brief Returns a new rectangle inset by the specified amounts along each axis.
 *
 * This function reduces the width and height of the given rectangle by twice the provided
 * inset amounts (`x_inset` and `y_inset`), ensuring that the resulting dimensions
 * are never negative. The rectangle remains centered at the same position as the original.
 *
 * @param rect The original rectangle to inset.
 * @param x_inset The amount to inset the rectangle on each side along the x-axis.
 * @param y_inset The amount to inset the rectangle on each side along the y-axis.
 * @return A new `Rectangle` instance representing the inset rectangle.
 *
 * @note If the inset amount exceeds half the original size, the corresponding dimension
 * will be clamped to zero.
 */
Rectangle inset_rectangle(const Rectangle &rect, float x_inset, float y_inset);

// TODO: there needs to be a scale from position function at some point.
/// @brief scales a rectangle uniformly by scale from its center point. Could we just add a transform to rect?
Rectangle scale_rectangle(const Rectangle &rect, float scale);
Rectangle scale_rectangle(const Rectangle &rect, float x_scale, float y_scale);
/// @brief scales a rectangle keeping the left side in place
Rectangle scale_rectangle_from_left_side(const Rectangle &rect, float shrink);
Rectangle scale_rectangle_from_left_side(const Rectangle &rect, float x_shrink, float y_shrink);
Rectangle scale_rectangle_from_left_side(const Rectangle &rect, float x_shrink, float y_shrink = 1);
Rectangle scale_rectangle_from_top_side(const Rectangle &rect, float x_shrink, float y_shrink);
Rectangle scale_rectangle_from_top_left(const Rectangle &rect, float x_shrink, float y_shrink);
Rectangle scale_rectangle_from_bottom_left(const Rectangle &rect, float x_shrink, float y_shrink);
/**
 * @brief Slides a rectangle by a given offset in terms of its width and height.
 *
 * This function creates a new rectangle based on the input rectangle and moves
 * its center by the specified offsets. The offsets are multiplied by the
 * rectangle's width and height respectively.
 *
 * @param rect The original rectangle to slide.
 * @param x_offset The horizontal offset, multiplied by the rectangle's width.
 * @param y_offset The vertical offset, multiplied by the rectangle's height.
 * @return Rectangle A new rectangle with its center moved by the specified offsets.
 */
Rectangle slide_rectangle(const Rectangle &rect, int x_offset, int y_offset);
Rectangle get_bounding_rectangle(const std::vector<Rectangle> &rectangles);

/**
 * @brief a rectangular grid which allows you to get the rectangles out of it
 *
 *
 */
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

/**
 * Binary Text Grid:
 *
 * ----------
 * -****-----
 * **--**----
 * **--**----
 * **--**----
 * **--**----
 * **--**----
 * **--**----
 * **--**----
 * -****-----
 * ----------
 * ----------
 * ----------
 *
 * Also note that "()" is how you can make multiline strings in c++
 *
 * Is string with newlines where each line has the exact same length forming a sort of 2d grid. The
 * elements in the grid are binary, an asterisk indicates that pixel is "on", and a space indicates that it is off.
 *
 * This function takes in a binary text grid, and then constructs an equivalent grid of squares wherever the binary text
 * grid is on
 *
 * The purpose of this function is to allow you to make simple pixel art or fonts without having to use images
 *
 */
draw_info::IndexedVertexPositions binary_text_grid_to_rect_grid(const std::string &text_grid,
                                                                const vertex_geometry::Rectangle bounding_rect);

/**
 * @brief Generates a triangle flat on the XY plane at Z = 0 with a configurable tip position.
 *
 * The triangle is centered at the origin (0,0,0). The tip of the triangle can be offset along
 * the X-axis of the base using `tip_offset` in the range [-1, 1]:
 * - `0` places the tip in the middle of the base (default),
 * - `1` aligns the tip to the right end of the base,
 * - `-1` aligns the tip to the left end of the base.
 *
 * @param width The width of the triangle's base along the X-axis.
 * @param height The vertical height of the triangle along the Y-axis.
 * @param tip_offset Optional. A value in [-1,1] specifying the tip's horizontal offset. Default is 0.0f.
 * @return draw_info::IndexedVertexPositions Contains the triangle's vertices and indices.
 */
draw_info::IndexedVertexPositions generate_triangle_with_tip_offset(float width, float height, float tip_offset = 0.0f);

/**
 * @brief Generates a right-angle triangle flat on the XY plane at Z = 0.
 *
 * The triangle is created by calling `generate_triangle_with_tip_offset` with `tip_offset` set
 * to `1` or `-1` depending on alignment:
 * - `positive_x_aligned = true` → right angle at bottom-left, tip at right (+X) side,
 * - `positive_x_aligned = false` → right angle at bottom-right, tip at left (-X) side.
 *
 * @param width The width of the triangle's base along the X-axis.
 * @param height The vertical height of the triangle along the Y-axis.
 * @param positive_x_aligned Optional. If true, the right angle is at the bottom-left. Default is true.
 * @return draw_info::IndexedVertexPositions Contains the triangle's vertices and indices.
 */
draw_info::IndexedVertexPositions generate_right_angle_triangle(float width, float height,
                                                                bool positive_x_aligned = true);

draw_info::IndexedVertexPositions generate_rectangle_between_2d(const glm::vec2 &p1, const glm::vec2 &p2,
                                                                float thickness = 0.01);

Rectangle create_rectangle_from_corners(const glm::vec3 top_left, const glm::vec3 top_right,
                                        const glm::vec3 bottom_left, const glm::vec3 bottom_right);
Rectangle create_rectangle(float x_pos, float y_pos, float width, float height);

/**
 * @brief Creates a rectangle from a specified reference point and size.
 *
 * These functions create a Rectangle by specifying a reference point on the rectangle
 * (such as a corner or an edge center) and the rectangle's width and height. The rectangle's
 * internal center position is computed based on the reference point:
 *
 * @param reference_point The reference point (corner or edge center) to position the rectangle.
 * @param width The width of the rectangle.
 * @param height The height of the rectangle.
 * @return Rectangle The constructed rectangle with its center at the calculated position.
 *
 * @note the naming convention is (top/bottom)_(left/right) or (left/right/top/bottom)_center
 */
Rectangle create_rectangle_from_top_left(const glm::vec3 &top_left, float width, float height);
Rectangle create_rectangle_from_top_right(const glm::vec3 &top_right, float width, float height);
Rectangle create_rectangle_from_bottom_left(const glm::vec3 &bottom_left, float width, float height);
Rectangle create_rectangle_from_bottom_right(const glm::vec3 &bottom_right, float width, float height);
Rectangle create_rectangle_from_left_center(const glm::vec3 &center_left, float width, float height);
Rectangle create_rectangle_from_top_center(const glm::vec3 &top_center, float width, float height);
Rectangle create_rectangle_from_bottom_center(const glm::vec3 &bottom_center, float width, float height);
Rectangle create_rectangle_from_right_center(const glm::vec3 &center_right, float width, float height);

Rectangle create_rectangle_from_center(const glm::vec3 &center, float width, float height);

enum class CutDirection {
    vertical,
    horizontal,
};

/**
 * @brief Subdivides a rectangle into a number of equally sized sub-rectangles.
 *
 * Depending on the `vertical` parameter, the rectangle is split either along the vertical axis:
 *
 * @code
 *
 * +------------------------+
 * |  0  |  1   |  2  |  3  |
 * +------------------------+
 *
 * @endcode
 *
 * (the cuts made are vertical)
 *
 * and when vertical is false we get horizontal cuts like this
 *
 * @code
 *
 * +---+
 * | 0 |
 * |---|
 * | 1 |
 * |---|
 * | 2 |
 * |---|
 * | 3 |
 * |---|
 * | 4 |
 * +---+
 *
 * @endcode
 *
 * @param rect The rectangle to subdivide.
 * @param num_subdivisions The number of sub-rectangles to create.
 * @param vertical If true, subdivision is vertical; otherwise, horizontal. Default is true.
 * @return A vector of `Rectangle` instances representing the subdivided regions.
 *
 * @todo create an enum called cut direction which is either vertical or horizontal and use that instead.
 */
std::vector<Rectangle> subdivide_rectangle(const Rectangle &rect, unsigned int num_subdivisions,
                                           CutDirection cut_direction = CutDirection::vertical);

/**
 * @brief Subdivides a rectangle vertically according to specified weights.
 *
 * Each weight in `weights` represents the relative height of a sub-rectangle. The resulting
 * sub-rectangles are stacked vertically, maintaining the same width as the original rectangle.
 *
 * @param rect The rectangle to subdivide.
 * @param weights A vector of weights defining the relative heights of each sub-rectangle.
 * @return A vector of `Rectangle` instances representing the vertically subdivided regions.
 *
 * @note Think of vertical subdivision as cutting lines of text in a book.
 */
std::vector<Rectangle> vertical_weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights);

/**
 * @brief Subdivides a rectangle horizontally according to specified weights.
 *
 * Each weight in `weights` represents the relative width of a sub-rectangle. The resulting
 * sub-rectangles are laid out horizontally, maintaining the same height as the original rectangle.
 *
 * @param rect The rectangle to subdivide.
 * @param weights A vector of weights defining the relative widths of each sub-rectangle.
 * @return A vector of `Rectangle` instances representing the horizontally subdivided regions.
 *
 * @note Think of horizontal subdivision as slicing a carrot on a cutting board.
 */
std::vector<Rectangle> horizontal_weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights);

/**
 * @brief Subdivides a rectangle according to specified weights, either vertically or horizontally.
 *
 * This function combines the behavior of `vertical_weighted_subdivision` and
 * `horizontal_weighted_subdivision` depending on the `vertical` flag. Each weight defines
 * the relative size of the corresponding sub-rectangle along the chosen axis.
 *
 * @param rect The rectangle to subdivide.
 * @param weights A vector of weights defining the relative sizes of sub-rectangles.
 * @param vertical If true, subdivision is vertical; otherwise, horizontal. Default is true.
 * @return A vector of `Rectangle` instances representing the weighted subdivided regions.
 */
std::vector<Rectangle> weighted_subdivision(const Rectangle &rect, const std::vector<unsigned int> &weights,
                                            CutDirection cut_direction = CutDirection::vertical);

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

/**
 * @brief Generates a wedge-shaped 3D prism centered at the origin.
 *
 * The wedge is constructed by extruding a triangle along the Z-axis. The triangle
 * lies on the XY plane, and the wedge extends symmetrically along Z so that the
 * prism is centered at the origin. The tip of the triangle can be horizontally
 * offset using `tip_offset`.
 *
 * @param width The width of the triangle's base along the X-axis. Default is 1.
 * @param height The height of the triangle along the Y-axis. Default is 1.
 * @param depth The depth of the wedge along the Z-axis. Default is 1.
 * @param tip_offset Optional. A value in [-1, 1] controlling the horizontal position
 *                   of the triangle's tip along the base:
 *
 *                   - 0.0 (default): tip is centered in the base
 *                   - 1.0 : tip aligned to the right end of the base,
 *                   - -1.0 : tip aligned to the left end of the base.
 *
 * @return draw_info::IndexedVertexPositions Contains the wedge's vertices and indices.
 */
draw_info::IndexedVertexPositions generate_wedge(float width = 1, float height = 1, float depth = 1,
                                                 float tip_offset = 0.0f);

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

draw_info::IndexedVertexPositions connect_points_by_rectangles(const std::vector<glm::vec2> &points);

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

// NOTE: this is only in a 2d flat sense, so idk if this was the best naming choice
draw_info::IndexedVertexPositions generate_rectangle(float center_x, float center_y, float width, float height);
std::vector<glm::vec3> generate_rectangle_vertices(float center_x, float center_y, float width, float height);
// TODO: the below shouldn't exist, instead the above should just take in z, but i don't want to bust the api right
// now
draw_info::IndexedVertexPositions generate_rectangle(float center_x, float center_y, float center_z, float width,
                                                     float height);
std::vector<glm::vec3> generate_rectangle_vertices_with_z(float center_x, float center_y, float center_z, float width,
                                                          float height);
/**
 * @brief Generates the index buffer for a rectangle made of two triangles.
 *
 * This function returns a std::vector of unsigned integers representing
 * the indices for drawing a rectangle (quad) using two triangles. The
 * vertex ordering assumes the rectangle vertices are numbered as follows:
 *
 *   0 --- 1
 *   |   / |
 *   |  /  |
 *   | /   |
 *   3 --- 2
 *
 * The triangles are defined as:
 * - First triangle: vertices 0, 1, 3
 * - Second triangle: vertices 1, 2, 3
 *
 * @return std::vector<unsigned int> The indices for the rectangle.
 */
std::vector<unsigned int> generate_rectangle_indices();
std::vector<glm::vec2> generate_rectangle_texture_coordinates();
std::vector<glm::vec2> generate_rectangle_texture_coordinates_flipped_vertically();

/**
 * @brief Generates the four corner vertices of a 3D rectangle (quad).
 *
 * Given a rectangle's center, orientation, and size, this function
 * computes the positions of its four corners in 3D space. The rectangle
 * is defined by two direction vectors: one for the width and one for the
 * height, and their respective magnitudes.
 *
 * The vertices are returned in the following order:
 * 1. Top right
 * 2. Bottom right
 * 3. Bottom left
 * 4. Top left
 *
 * @param center The center position of the rectangle in 3D space.
 * @param width_dir The direction vector along the rectangle's width.
 * @param height_dir The direction vector along the rectangle's height.
 * @param width The total width of the rectangle.
 * @param height The total height of the rectangle.
 * @return std::vector<glm::vec3> A vector containing the four corner vertices.
 *
 * @note the width dir and height dir could also just store their magnitudes directly.
 */
std::vector<glm::vec3> generate_rectangle_vertices_3d(const glm::vec3 &center, const glm::vec3 &width_dir,
                                                      const glm::vec3 &height_dir, float width, float height);

draw_info::IndexedVertexPositions generate_rectangle_3d(const glm::vec3 &center, const glm::vec3 &width_dir,
                                                        const glm::vec3 &height_dir, float width, float height);

std::vector<glm::vec3> generate_rectangle_vertices_from_points(const glm::vec3 &point_a, const glm::vec3 &point_b,
                                                               const glm::vec3 &surface_normal, float height = 1);

/**
 * @brief generate axes showing x y and
 *
 */

draw_info::IndexedVertexPositions generate_3d_arrow_with_ratio(const glm::vec3 &start, const glm::vec3 &end,
                                                               int num_segments = 16,
                                                               float length_thickness_ratio = 0.07);

draw_info::IndexedVertexPositions generate_3d_axes();

draw_info::IndexedVertexPositions generate_3d_arrow(const glm::vec3 &start, const glm::vec3 &end, int num_segments = 16,
                                                    float stem_thickness = 0.12);

draw_info::IndexedVertexPositions generate_2d_arrow(glm::vec2 start, glm::vec2 end, float stem_thickness = 0.05,
                                                    float tip_length = .05);

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

draw_info::IndexedVertexPositions generate_star(float center_x, float center_y, float outer_radius, float inner_radius,
                                                int num_star_tips, bool blunt_tips = false);
std::vector<glm::vec3> generate_star_vertices(float center_x, float center_y, float outer_radius, float inner_radius,
                                              int num_star_tips, bool blunt_tips = false);
std::vector<unsigned int> generate_star_indices(int num_star_tips, bool blunt_tips);

std::vector<glm::vec3> generate_fibonacci_sphere_vertices(int num_samples, float scale);

void translate_vertices_in_place(std::vector<glm::vec3> &vertices, const glm::vec3 &translation);
void increment_indices_in_place(std::vector<unsigned int> &indices, unsigned int increase);
} // namespace vertex_geometry

#endif // VERTEX_GEOMETRY_HPP
