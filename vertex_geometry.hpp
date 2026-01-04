#ifndef VERTEX_GEOMETRY_HPP
#define VERTEX_GEOMETRY_HPP

#include <glm/glm.hpp>
#include <type_traits>
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
    // Construct from explicit points, also the points must be planar...
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

        if (normal == glm_utils::zero_R3)
            throw std::runtime_error("Normal must be non-zero");

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

    AxisAlignedBoundingBox(const glm::vec3 &min, const glm::vec3 &max)
        : vertex_geometry::AxisAlignedBoundingBox(std::vector{min, max}) {}

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

    double volume() const { return (max.x - min.x) * (max.y - min.y) * (max.z - min.z); }

    std::optional<AxisAlignedBoundingBox> intersection(const AxisAlignedBoundingBox &other) const {
        double ix_min = std::max(min.x, other.min.x);
        double iy_min = std::max(min.y, other.min.y);
        double iz_min = std::max(min.z, other.min.z);

        double ix_max = std::min(max.x, other.max.x);
        double iy_max = std::min(max.y, other.max.y);
        double iz_max = std::min(max.z, other.max.z);

        if (ix_min <= ix_max && iy_min <= iy_max && iz_min <= iz_max) {
            return AxisAlignedBoundingBox{{ix_min, iy_min, iz_min}, {ix_max, iy_max, iz_max}};
        } else {
            return std::nullopt;
        }
    }

    /**
     * @brief Converts the bounding box into indexed vertex positions for drawing.
     * @return A draw_info::IndexedVertexPositions object representing the box geometry.
     */
    draw_info::IndexedVertexPositions get_ivp() const;
};

enum class ExtentMode { half, full };

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

    AxisAlignedBoundingBox2D() {
        min = glm::vec2(std::numeric_limits<float>::max());
        max = glm::vec2(std::numeric_limits<float>::lowest());
    }

    AxisAlignedBoundingBox2D(const glm::vec2 &min, const glm::vec2 &max)
        : vertex_geometry::AxisAlignedBoundingBox2D(std::vector{min, max}) {}

    // constructs aabb from extent, centered at origin
    AxisAlignedBoundingBox2D(const glm::vec2 &extent, ExtentMode extent_mode = ExtentMode::full)
        : vertex_geometry::AxisAlignedBoundingBox2D(-extent * ((extent_mode == ExtentMode::full) ? 0.5f : 1.0f),
                                                    extent * ((extent_mode == ExtentMode::full) ? 0.5f : 1.0f)) {}

    // constructs aabb from extents, centered anywhere
    AxisAlignedBoundingBox2D(const glm::vec2 &center, float x_extent, float y_extent,
                             ExtentMode extent_mode = ExtentMode::full)
        : vertex_geometry::AxisAlignedBoundingBox2D(
              center + glm_utils::x_R2 * ((extent_mode == ExtentMode::full) ? 0.5f : 1.0f) * x_extent,
              center + glm_utils::y_R2 * ((extent_mode == ExtentMode::full) ? 0.5f : 1.0f) * y_extent) {}

    // pushing directionally out.
    AxisAlignedBoundingBox2D(const glm::vec2 &center, float x_extent_pos, float x_extent_neg, float y_extent_pos,
                             float y_extent_neg)
        : vertex_geometry::AxisAlignedBoundingBox2D(center + glm::vec2(-x_extent_neg, -y_extent_neg),
                                                    center + glm::vec2(x_extent_pos, y_extent_pos)) {}

    glm::vec2 min;
    glm::vec2 max;

    AxisAlignedBoundingBox2D centered_at_origin() const {
        const glm::vec2 half_extents = (max - min) * 0.5f;
        return AxisAlignedBoundingBox2D{-half_extents, half_extents};
    }

    glm::vec2 get_center() const { return (min + max) * 0.5f; }
    float get_x_size() const { return (max.x - min.x); }
    float get_y_size() const { return (max.y - min.y); }

    // returns true if this aabb contains the other one
    bool contains(const vertex_geometry::AxisAlignedBoundingBox2D &other) const {
        return other.min.x >= min.x && other.min.y >= min.y && other.max.x <= max.x && other.max.y <= max.y;
    }

    // Returns the 4 corners of the 2D bounding box
    std::array<glm::vec2, 4> get_corners() const {
        return {glm::vec2(min.x, min.y), glm::vec2(max.x, min.y), glm::vec2(min.x, max.y), glm::vec2(max.x, max.y)};
    }

    // LATER
    // draw_info::IndexedVertexPositions get_ivp();
};

std::vector<AxisAlignedBoundingBox> subtract_aabb(const AxisAlignedBoundingBox &A, const AxisAlignedBoundingBox &B);

// warning, can be exponentialy in the size of B
std::vector<AxisAlignedBoundingBox> subtract_aabbs(const AxisAlignedBoundingBox &A,
                                                   const std::vector<AxisAlignedBoundingBox> &Bs);

/**
 * @brief Subtracts one axis-aligned bounding box (AABB) from another.
 *
 * Given a base AABB `A` and an overlapping AABB `B`, this function computes
 * the remaining regions of `A` after "cutting out" `B`. The result is a
 * collection of 0 to 4 disjoint AABBs representing the leftover space.
 *
 * The possible resulting regions are:
 * - Left strip of A left of B
 * - Right strip of A right of B
 * - Bottom strip of A below B
 * - Top strip of A above B
 *
 * If `B` does not overlap `A`, the result contains only `A`.
 * If `B` completely covers `A`, the result is empty.
 *
 * @param A The base AABB to subtract from.
 * @param B The AABB to subtract.
 * @return std::vector<AxisAlignedBoundingBox2D> A vector of remaining AABBs.
 */
std::vector<AxisAlignedBoundingBox2D> subtract_aabb(const AxisAlignedBoundingBox2D &A,
                                                    const AxisAlignedBoundingBox2D &B);

draw_info::IndexedVertexPositions triangulate(const std::vector<glm::vec3> &pts,
                                              TriangulationMode triangulation_mode = TriangulationMode::CentralFan);
draw_info::IndexedVertexPositions triangulate_ngon(const NGon &ngon);
draw_info::IndexedVertexPositions connect_ngons(const NGon &a, const NGon &b);

/**
 * @brief Specifies how size parameters are interpreted for geometric shapes.
 *
 * This enum is used to disambiguate constructors or functions that accept
 * size parameters for a Rectangle (or other geometric objects).
 *
 * An example in the context of a rectangle where the provided `u` and `v` values represent half-extents or full
 * extents.
 *
 * - `Extent::half` : Values represent the half-length along each axis. For example, a
 *   `u_size` of 1 means the rectangle extends from -1 to 1 along the u-axis (length 2).
 * - `Extent::full` : Values represent the full length along each axis. For example, a
 *   `u_size` of 2 means the rectangle extends from -1 to 1 along the u-axis (length 2).
 *
 * Most of the time you use half extents when you
 *
 * @code
 * // Example: create a rectangle using half-extents
 * Rectangle r1({0, 0, 0}, 1.0f, 2.0f, Extent::half);
 * // u-size = 1 (half-length), v-size = 2 (half-length)
 *
 * // Example: create a rectangle using full dimensions
 * Rectangle r2({0, 0, 0}, 2.0f, 4.0f, Extent::full);
 * // u-size = 2 (full-length), v-size = 4 (full-length)
 *
 * // Internally, both constructors store half-extents for consistent representation
 * @endcode
 *
 * @note Using this enum makes constructor intent explicit and prevents accidental
 * misuse of size parameters.
 *
 * @note The biggest thing to realize is that for some constant c, a geometric object with half-extent equal to c,
 * will be twice as large as the one with the full extent equal to c
 */

enum class Plane {
    XY,
    XZ,
    YZ,
    YX = XY, // optional alias
    ZX = XZ, // optional alias
    ZY = YZ  // optional alias
};

// TODO: would it be beneficial to have a Rectangle2D class as well?
class Rectangle {

  public:
    ExtentMode extent_mode = ExtentMode::half;

  private:
    // u and v are vectors that go from the center to the edge of the rectangle, and are definitionally half-extents
    // TODO: these probably don't have to be private anymore, but we need setters for them.
    glm::vec3 u;
    glm::vec3 v;

  public:
    /**
     * @brief Constructs a Rectangle given a center and two edge vectors. A rectangle has corners with angles all
     * equal to 90 degrees, and thus is a subset of all possible quads.
     *
     * @note If the input `v` is not perpendicular to `u`, it will be automatically
     *       orthogonalized. This ensures that the resulting rectangle is a true rectangle
     *       (not a parallelogram), but the final direction of `v` may differ from the original input.
     *
     * @param center The center position of the rectangle.
     * @param u Vector from the center to one edge along the u axis .
     * @param v Vector from the center to one edge along the v axis (should be perp to u).
     */
    Rectangle(glm::vec3 center, glm::vec3 u, glm::vec3 v, ExtentMode extent_mode = ExtentMode::half)
        : center(center), extent_mode(extent_mode) {

        this->u = u;

        // make v perpendicular to u if its not
        glm::vec3 u_dir = glm::normalize(u);           // direction of u
        glm::vec3 v_proj = glm::dot(v, u_dir) * u_dir; // projection of v onto u
        this->v = v - v_proj;                          // perpendicular component

        // preserve the original length of v
        float v_original_length = glm::length(v);
        this->v = glm::normalize(this->v) * v_original_length;
    }

    // by default we use width height 2 to take up the full [-1, 1] x [-1, 1] ndc space
    // NOTE: by default the rectangle will be planar with the XY plane
    // TODO: make it take in a plane to go along ?
    // NOTE: full extents were chosen here because of the fact that many other things rely on this being in full
    // extents, sometime in the future it would be better for that other code, to rely on an an AABB(2D) and go back
    // to talking about width and height rather than using generic retangles.
    Rectangle(glm::vec3 center = glm::vec3(0), float u_size = 2, float v_size = 2,
              ExtentMode extent_mode = ExtentMode::full)
        : Rectangle(center, (extent_mode == ExtentMode::half ? u_size : u_size * 0.5f) * glm_utils::x,
                    (extent_mode == ExtentMode::half ? v_size : v_size * 0.5f) * glm_utils::y, extent_mode) {}

    // TODO: make private and use setters
    glm::vec3 center; // Center position

    // TODO: in the future we can use lazily computed values using Mutable access notification for all dependencies.

    float get_u_extent_size() const {
        if (extent_mode == ExtentMode::half) {
            return glm::length(u); // half-extent mode: internal length matches user
        } else {
            return glm::length(u) * 2.0f; // full-extent mode: internal is half, multiply by 2
        }
    }

    float get_v_extent_size() const {
        if (extent_mode == ExtentMode::half) {
            return glm::length(v);
        } else {
            return glm::length(v) * 2.0f;
        }
    }

    void set_u_extent(float new_u_size) {
        float factor = (extent_mode == ExtentMode::half) ? 1.0f : 0.5f;
        if (glm::length(u) > 0.0f) {
            u = glm::normalize(u) * new_u_size * factor;
        } else {
            u = glm_utils::x * new_u_size * factor;
        }
    }

    void set_v_extent(float new_v_size) {
        float factor = (extent_mode == ExtentMode::half) ? 1.0f : 0.5f;
        if (glm::length(v) > 0.0f) {
            v = glm::normalize(v) * new_v_size * factor;
        } else {
            v = glm_utils::y * new_v_size * factor;
        }
    }

    // TODO: should I extract this out so this class has nothing to do with draw info?
    draw_info::IndexedVertexPositions get_ivp() const;
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

// TODO: in the future when required we can add the option to pass a plane to be planar to or something like that.
Rectangle aabb2d_to_rect(const AxisAlignedBoundingBox2D &aabb2d);

// TODO: I'm pretty sure all the below logic is for Rectangle2D's, but it works with the current rectangle so
// leaving it.

/**
 * @brief Returns a new rectangle with expanded edge-to-edge size along its local axes.
 *
 * This function creates a copy of the input rectangle and increases its
 * extents along the local `u` and `v` axes by the specified amounts.
 * The expansion is measured in terms of edge-to-edge size.
 * - orig_u, orig_v: original edge to edge measurment of rect along u, v axes respectively
 * The new edge to edge size is given by:
 * - orig_u + 2 * u_addition, orig_v + 2 * v_addition
 *
 * Geometrically this modification is made by expanding the edges along the u, v axes respectively by the specified
 * amount in each direction
 *
 * @param rect The original rectangle to expand.
 * @param u_addition The amount to add to the edge-to-edge size along the rectangle's `u` axis.
 * @param v_addition The amount to add to the edge-to-edge size along the rectangle's `v` axis.
 *
 * @return A new Rectangle with the same center as `rect` but with expanded extents along the `u` and `v` axes.
 *
 * @note The original rectangle `rect` is not modified.
 */
Rectangle expand_rectangle(const Rectangle &rect, float u_addition, float v_addition);

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
Rectangle scale_rectangle_from_top_right(const Rectangle &rect, float x_shrink, float y_shrink);
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
    std::vector<Rectangle> get_rectangles_in_bounding_box(int col1, int row1, int col2, int row2) const;
    std::vector<Rectangle> get_row(int row) const;
    std::vector<Rectangle> get_column(int col) const;

    const int rows; // Number of rows in the grid
    const int cols; // Number of columns in the grid

    std::vector<Rectangle> get_all_rectangles() const {
        std::vector<Rectangle> all_rects;
        all_rects.reserve(rows * cols); // avoid reallocations

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                all_rects.push_back(get_at(c, r));
            }
        }

        return all_rects;
    }

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
 * This function takes in a binary text grid, and then constructs an equivalent grid of squares wherever the binary
 * text grid is on
 *
 * The purpose of this function is to allow you to make simple pixel art or fonts without having to use images
 *
 */

std::vector<draw_info::IndexedVertexPositions>
binary_text_grid_to_rect_grid_split(const std::string &text_grid, const vertex_geometry::Rectangle bounding_rect);
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

// TODO: these all need to be renamed to quad, because they're not rectangles!
draw_info::IndexedVertexPositions generate_rectangle_between_2d(const glm::vec2 &p1, const glm::vec2 &p2,
                                                                float thickness = 0.01, Plane plane = Plane::XZ);

draw_info::IndexedVertexPositions generate_rectangle_between(const glm::vec3 &p1, const glm::vec3 &p2, float thickness,
                                                             const glm::vec3 &normal = glm_utils::y);

draw_info::IndexedVertexPositions connect_points_by_rectangles_2d(const std::vector<glm::vec2> &points,
                                                                  float thickness = 0.01, Plane plane = Plane::XZ);

/**
 * @brief Generates a sequence of rectangles connecting consecutive 3D points.
 *
 * Each rectangle is represented as a draw_info::IndexedVertexPositions object.
 * Optionally, the rectangles can share edges so that consecutive rectangles
 * have perfectly aligned edges, avoiding cracks or overlap in the rectangles.
 *
 * @param points A vector of glm::vec3 points to connect.
 *               Must contain at least two points.
 * @param thickness The thickness of each rectangle (default: 0.01).
 * @param normal The normal vector used to compute the perpendicular direction
 *               for the rectangle (default: glm_utils::y).
 * @param share_edges If true, consecutive rectangles share edges to align perfectly
 *                    (default: true).
 *
 * @return std::vector<draw_info::IndexedVertexPositions> A vector of rectangles,
 *         one for each segment between consecutive points.
 */
std::vector<draw_info::IndexedVertexPositions>
connect_points_by_rectangles_split(const std::vector<glm::vec3> &points, float thickness = 0.01,
                                   const glm::vec3 &normal = glm_utils::y, bool share_edges = true);

/**
 * @brief Connects a sequence of 3D points by rectangles and merges them into a single mesh.
 *
 * This function internally calls connect_points_by_rectangles_split() and then merges
 * all resulting rectangles into one draw_info::IndexedVertexPositions object,
 * suitable for rendering as a single mesh.
 *
 * @param points A vector of glm::vec3 points to connect.
 *               Must contain at least two points.
 * @param thickness The thickness of each rectangle (default: 0.01).
 * @param normal The normal vector used to compute the perpendicular direction
 *               for the rectangle (default: glm_utils::y).
 * @param share_edges If true, consecutive rectangles share edges to align perfectly
 *                    (default: true).
 *
 * @return draw_info::IndexedVertexPositions A single merged mesh representing all rectangles.
 */
draw_info::IndexedVertexPositions connect_points_by_rectangles(const std::vector<glm::vec3> &points,
                                                               float thickness = 0.01,
                                                               const glm::vec3 &normal = glm_utils::y,
                                                               bool share_edges = true);

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
    /*
     * @code
     * +-----------------------+
     * |     |     |     |     |
     * |     |     |     |     |
     * |  0  |  1  |  2  |  3  |
     * |     |     |     |     |
     * |     |     |     |     |
     * +-----------------------+
     * @endcode
     */
    vertical,
    /*
     * @code
     * +-----------------------+
     * |           0           |
     * |-----------------------|
     * |           1           |
     * |-----------------------|
     * |           2           |
     * |-----------------------|
     * |           3           |
     * +-----------------------+
     * @endcode
     */
    horizontal,
};

/**
 * @brief Subdivides a rectangle into a number of equally sized sub-rectangles.
 *
 * Depending on the `vertical` parameter, the rectangle is split either along the vertical axis:
 *
 *
 * (the cuts made are vertical)
 *
 * and when vertical is false we get horizontal cuts like this
 *
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

// TODO: use perlin noise which is in math utils to actually do this cleanly
draw_info::IVPNormals generate_terrain(float size_x = 100.0f, float size_z = 100.0f, int resolution_x = 50,
                                       int resolution_z = 50, float max_height = 5.0f, float base_height = 0.0f,
                                       int octaves = 4, float persistence = 0.5f, float scale = 50.0f,
                                       float seed = 0.0f);

draw_info::IndexedVertexPositions connect_points_by_rectangles(const std::vector<glm::vec2> &points);

/**
 * @brief Generates a 3D cylinder mesh visualizing a parametric function.
 *
 * This function samples a parametric curve `f(t)` over the interval `[t_start, t_end]`
 * using a specified `step_size` and computes tangent vectors via finite differences.
 * It then constructs a segmented cylinder along the sampled points to visualize the curve in 3D.
 *
 * @param f The parametric function to visualize. It takes a `double t` and returns a 3D point (`glm::vec3`).
 * @param t_start The starting value of the parameter `t`.
 * @param t_end The ending value of the parameter `t`.
 * @param step_size The increment in `t` between consecutive samples.
 * @param finite_diff_delta A small delta used for approximating the tangent vector via finite differences.
 * @param radius The radius of the cylinder to generate along the curve.
 * @param segments The number of vertices per circular cross-section of the cylinder.
 *
 * @note This function internally uses `sample_points_and_tangents` to sample the curve
 *       and `generate_segmented_cylinder` to generate the mesh. At least two sample points
 *       are required for a valid cylinder mesh.
 */
draw_info::IndexedVertexPositions generate_function_visualization(std::function<glm::vec3(double)> f, double t_start,
                                                                  double t_end, double step_size,
                                                                  double finite_diff_delta, float radius = .25,
                                                                  int segments = 8);

/**
 * @brief Generates a segmented cylinder mesh along a given 3D path.
 *
 * This function constructs a cylindrical mesh that follows the 3D curve defined by `path`.
 * The cylinder is built by connecting consecutive points in the path with circular cross-sections.
 * Each segment of the cylinder is approximated using `segments` vertices per ring, forming triangles
 * to create a closed surface.
 *
 * @param path A vector of pairs representing the curve. Each pair contains:
 *             - first: the position of a point on the path (`glm::vec3`)
 *             - second: the tangent at that point (`glm::vec3`) [not used in this function, can be used externally]
 * @param radius The radius of the cylinder.
 * @param segments The number of vertices per circular cross-section (higher values produce smoother cylinders).
 *
 * @note The function uses the Gram-Schmidt process to compute an arbitrary perpendicular frame (normal and
 * binormal) for each segment to construct the circular cross-sections. At least two points in `path` are required;
 *       otherwise, an empty mesh is returned.
 */
draw_info::IndexedVertexPositions generate_segmented_cylinder(const std::vector<std::pair<glm::vec3, glm::vec3>> &path,
                                                              float radius, int segments);

draw_info::IndexedVertexPositions generate_segmented_cylinder(const std::vector<glm::vec3> &path, float radius,
                                                              int segments);

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

// NOTE: this should replace all the above rectangles, they are bad.
draw_info::IndexedVertexPositions generate_rectangle(const glm::vec3 &center, const glm::vec3 &u = glm_utils::x,
                                                     const glm::vec3 &v = glm_utils::z);

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
 * @brief Generates a 3D arrow between two points with a thickness defined by a length-to-thickness ratio.
 *
 * This function computes a 3D arrow from @p start to @p end by determining the arrow's stem thickness
 * as a fixed ratio of the total arrow length. The computed thickness is then forwarded to
 * generate_3d_arrow(), which performs the actual geometry construction.
 *
 * @param start The starting point of the arrow in 3D space.
 * @param end The end point of the arrow in 3D space.
 * @param num_segments The number of radial subdivisions used when generating the arrow’s cylindrical
 *                     and conical components.
 * @param length_thickness_ratio The ratio of arrow length to stem thickness. For example,
 *                               a value of 0.05 makes the stem thickness 5% of the arrow length.
 *
 * @return A draw_info::IndexedVertexPositions object containing the indexed vertices and topology
 *         describing the final arrow geometry.
 *
 *
 * @see generate_3d_arrow()
 */
draw_info::IndexedVertexPositions generate_3d_arrow_with_ratio(const glm::vec3 &start, const glm::vec3 &end,
                                                               int num_segments = 16,
                                                               float length_thickness_ratio = 0.07);

draw_info::IndexedVertexPositions generate_3d_axes();

/**
 * @brief Generates a full 3D arrow mesh between two points with a specified stem thickness.
 *
 * This function constructs a complete arrow consisting of:
 *   - a cylindrical stem running from @p start to the base of the tip, and
 *   - a conical tip reaching from the end of the stem to @p end.
 *
 * The tip length is defined as 30% of the total arrow length, and the tip base radius is twice the
 * stem thickness. The function generates geometry for both components and merges them into a single
 * IndexedVertexPositions structure.
 *
 * @param start The starting point of the arrow in 3D space.
 * @param end The endpoint of the arrow tip.
 * @param num_segments The number of radial segments used for the cylinder and cone. Higher values
 *                     result in smoother geometry.
 * @param stem_thickness The radius (thickness) of the cylindrical stem portion of the arrow.
 *
 * @return A draw_info::IndexedVertexPositions object representing the complete arrow mesh, including
 *         both stem and tip geometry.
 *
 * @see generate_3d_arrow_with_ratio()
 * @see vertex_geometry::generate_cylinder_between()
 * @see vertex_geometry::generate_cone_between()
 *
 * @note this is probably deprecated in favor of the generate_3d_arrow_with_ratio
 */
draw_info::IndexedVertexPositions generate_3d_arrow(const glm::vec3 &start, const glm::vec3 &end, int num_segments = 16,
                                                    float stem_thickness = 0.12);

draw_info::IndexedVertexPositions generate_2d_arrow(glm::vec2 start, glm::vec2 end, float stem_thickness = 0.05,
                                                    float tip_length = .05);

/**
 * @brief Generates a combined 3D arrow path where each arrow points
 *        from one point to the next in the given sequence.
 *
 * This function iterates over the provided @p points and constructs
 * a 3D arrow for every consecutive pair of points using
 * generate_3d_arrow_with_ratio(). The resulting individual arrow
 * meshes are then combined into a single
 * draw_info::IndexedVertexPositions structure using merge_ivps().
 *
 * If the input contains fewer than two points, an empty geometry
 * (default-constructed IndexedVertexPositions) is returned.
 *
 * @param points A list of 3D positions through which the arrows should be drawn.
 * @param num_segments Number of segments used for cylindrical portions of each arrow.
 * @param length_thickness_ratio Thickness ratio used for arrow stem relative to arrow length.
 *
 * @return A merged IndexedVertexPositions containing all arrows along the path.
 */
draw_info::IndexedVertexPositions generate_arrow_path(const std::vector<glm::vec3> &points, int num_segments = 16,
                                                      float length_thickness_ratio = 0.07f);

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
