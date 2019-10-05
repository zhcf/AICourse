// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "opaque_types.h"
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

template <typename T>
line ht_get_line (
    const hough_transform& ht,
    const dlib::vector<T,2>& p
)  
{ 
    DLIB_CASSERT(get_rect(ht).contains(p));
    auto temp = ht.get_line(p); 
    return line(temp.first, temp.second);
}

template <typename T>
double ht_get_line_angle_in_degrees (
    const hough_transform& ht,
    const dlib::vector<T,2>& p 
)  
{ 
    DLIB_CASSERT(get_rect(ht).contains(p));
    return ht.get_line_angle_in_degrees(p); 
}

template <typename T>
py::tuple ht_get_line_properties (
    const hough_transform& ht,
    const dlib::vector<T,2>& p
)  
{ 
    DLIB_CASSERT(get_rect(ht).contains(p));
    double angle_in_degrees;
    double radius;
    ht.get_line_properties(p, angle_in_degrees, radius);
    return py::make_tuple(angle_in_degrees, radius);
}

point ht_get_best_hough_point (
    hough_transform& ht,
    const point& p,
    const numpy_image<float>& himg
) 
{ 
    DLIB_CASSERT(num_rows(himg) == ht.size() && num_columns(himg) == ht.size() &&
        get_rect(ht).contains(p) == true,
        "\t point hough_transform::get_best_hough_point()"
        << "\n\t Invalid arguments given to this function."
        << "\n\t num_rows(himg): " << num_rows(himg)
        << "\n\t num_columns(himg): " << num_columns(himg)
        << "\n\t size():    " << ht.size()
        << "\n\t p:         " << p 
    );
    return ht.get_best_hough_point(p,himg); 
}

template <
    typename T 
    >
numpy_image<float> compute_ht (
    const hough_transform& ht,
    const numpy_image<T>& img,
    const rectangle& box
) 
{
    numpy_image<float> out;
    ht(img, box, out);
    return out;
}

template <
    typename T 
    >
numpy_image<float> compute_ht2 (
    const hough_transform& ht,
    const numpy_image<T>& img
) 
{
    numpy_image<float> out;
    ht(img, out);
    return out;
}

template <
    typename T 
    >
py::list ht_find_pixels_voting_for_lines (
    const hough_transform& ht,
    const numpy_image<T>& img,
    const rectangle& box,
    const std::vector<point>& hough_points,
    const unsigned long angle_window_size = 1,
    const unsigned long radius_window_size = 1
) 
{
    return vector_to_python_list(ht.find_pixels_voting_for_lines(img, box, hough_points, angle_window_size, radius_window_size));
}

template <
    typename T 
    >
py::list ht_find_pixels_voting_for_lines2 (
    const hough_transform& ht,
    const numpy_image<T>& img,
    const std::vector<point>& hough_points,
    const unsigned long angle_window_size = 1,
    const unsigned long radius_window_size = 1
) 
{
    return vector_to_python_list(ht.find_pixels_voting_for_lines(img, hough_points, angle_window_size, radius_window_size));
}

std::vector<point> ht_find_strong_hough_points(
    hough_transform& ht,
    const numpy_image<float>& himg,
    const float hough_count_thresh,
    const double angle_nms_thresh,
    const double radius_nms_thresh
)
{
    return ht.find_strong_hough_points(himg, hough_count_thresh, angle_nms_thresh, radius_nms_thresh);
}


// ----------------------------------------------------------------------------------------

void register_hough_transform(py::module& m)
{
    const char* class_docs =
"This object is a tool for computing the line finding version of the Hough transform \n\
given some kind of edge detection image as input.  It also allows the edge pixels \n\
to be weighted such that higher weighted edge pixels contribute correspondingly \n\
more to the output of the Hough transform, allowing stronger edges to create \n\
correspondingly stronger line detections in the final Hough transform.";


    const char* doc_constr = 
"requires \n\
    - size_ > 0 \n\
ensures \n\
    - This object will compute Hough transforms that are size_ by size_ pixels.   \n\
      This is in terms of both the Hough accumulator array size as well as the \n\
      input image size. \n\
    - size() == size_";
        /*!
            requires
                - size_ > 0
            ensures
                - This object will compute Hough transforms that are size_ by size_ pixels.  
                  This is in terms of both the Hough accumulator array size as well as the
                  input image size.
                - size() == size_
        !*/

    py::class_<hough_transform>(m, "hough_transform", class_docs)
        .def(py::init<unsigned long>(), doc_constr, py::arg("size_"))
        .def_property_readonly("size", &hough_transform::size,
            "returns the size of the Hough transforms generated by this object.  In particular, this object creates Hough transform images that are size by size pixels in size.")
        .def("get_line", &ht_get_line<long>, py::arg("p"))
        .def("get_line", &ht_get_line<double>, py::arg("p"),
"requires \n\
    - rectangle(0,0,size-1,size-1).contains(p) == true \n\
      (i.e. p must be a point inside the Hough accumulator array) \n\
ensures \n\
    - returns the line segment in the original image space corresponding \n\
      to Hough transform point p.  \n\
    - The returned points are inside rectangle(0,0,size-1,size-1).") 
    /*!
        requires
            - rectangle(0,0,size-1,size-1).contains(p) == true
              (i.e. p must be a point inside the Hough accumulator array)
        ensures
            - returns the line segment in the original image space corresponding
              to Hough transform point p. 
            - The returned points are inside rectangle(0,0,size-1,size-1).
    !*/

        .def("get_line_angle_in_degrees", &ht_get_line_angle_in_degrees<long>, py::arg("p"))
        .def("get_line_angle_in_degrees", &ht_get_line_angle_in_degrees<double>, py::arg("p"),
"requires \n\
    - rectangle(0,0,size-1,size-1).contains(p) == true \n\
      (i.e. p must be a point inside the Hough accumulator array) \n\
ensures \n\
    - returns the angle, in degrees, of the line corresponding to the Hough \n\
      transform point p.")
    /*!
        requires
            - rectangle(0,0,size-1,size-1).contains(p) == true
              (i.e. p must be a point inside the Hough accumulator array)
        ensures
            - returns the angle, in degrees, of the line corresponding to the Hough
              transform point p.
    !*/


        .def("get_line_properties", &ht_get_line_properties<long>, py::arg("p"))
        .def("get_line_properties", &ht_get_line_properties<double>, py::arg("p"),
"requires \n\
    - rectangle(0,0,size-1,size-1).contains(p) == true \n\
      (i.e. p must be a point inside the Hough accumulator array) \n\
ensures \n\
    - Converts a point in the Hough transform space into an angle, in degrees, \n\
      and a radius, measured in pixels from the center of the input image. \n\
    - let ANGLE_IN_DEGREES == the angle of the line corresponding to the Hough \n\
      transform point p.  Moreover: -90 <= ANGLE_IN_DEGREES < 90. \n\
    - RADIUS == the distance from the center of the input image, measured in \n\
      pixels, and the line corresponding to the Hough transform point p. \n\
      Moreover: -sqrt(size*size/2) <= RADIUS <= sqrt(size*size/2) \n\
    - returns a tuple of (ANGLE_IN_DEGREES, RADIUS)" )
    /*!
        requires
            - rectangle(0,0,size-1,size-1).contains(p) == true
              (i.e. p must be a point inside the Hough accumulator array)
        ensures
            - Converts a point in the Hough transform space into an angle, in degrees,
              and a radius, measured in pixels from the center of the input image.
            - let ANGLE_IN_DEGREES == the angle of the line corresponding to the Hough
              transform point p.  Moreover: -90 <= ANGLE_IN_DEGREES < 90.
            - RADIUS == the distance from the center of the input image, measured in
              pixels, and the line corresponding to the Hough transform point p.
              Moreover: -sqrt(size*size/2) <= RADIUS <= sqrt(size*size/2)
            - returns a tuple of (ANGLE_IN_DEGREES, RADIUS)
    !*/

        .def("get_best_hough_point", &ht_get_best_hough_point, py::arg("p"), py::arg("himg"),
"requires \n\
    - himg has size rows and columns. \n\
    - rectangle(0,0,size-1,size-1).contains(p) == true \n\
ensures \n\
    - This function interprets himg as a Hough image and p as a point in the \n\
      original image space.  Given this, it finds the maximum scoring line that \n\
      passes though p.  That is, it checks all the Hough accumulator bins in \n\
      himg corresponding to lines though p and returns the location with the \n\
      largest score.   \n\
    - returns a point X such that get_rect(himg).contains(X) == true")
    /*!
        requires
            - himg has size rows and columns.
            - rectangle(0,0,size-1,size-1).contains(p) == true
        ensures
            - This function interprets himg as a Hough image and p as a point in the
              original image space.  Given this, it finds the maximum scoring line that
              passes though p.  That is, it checks all the Hough accumulator bins in
              himg corresponding to lines though p and returns the location with the
              largest score.  
            - returns a point X such that get_rect(himg).contains(X) == true
    !*/

        .def("__call__", &compute_ht<uint8_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<uint16_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<uint32_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<uint64_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<int8_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<int16_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<int32_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<int64_t>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<float>, py::arg("img"), py::arg("box"))
        .def("__call__", &compute_ht<double>, py::arg("img"), py::arg("box"),
"requires \n\
    - box.width() == size \n\
    - box.height() == size \n\
ensures \n\
    - Computes the Hough transform of the part of img contained within box. \n\
      In particular, we do a grayscale version of the Hough transform where any \n\
      non-zero pixel in img is treated as a potential component of a line and \n\
      accumulated into the returned Hough accumulator image.  However, rather than \n\
      adding 1 to each relevant accumulator bin we add the value of the pixel \n\
      in img to each Hough accumulator bin.  This means that, if all the \n\
      pixels in img are 0 or 1 then this routine performs a normal Hough \n\
      transform.  However, if some pixels have larger values then they will be \n\
      weighted correspondingly more in the resulting Hough transform. \n\
    - The returned hough transform image will be size rows by size columns. \n\
    - The returned image is the Hough transform of the part of img contained in \n\
      box.  Each point in the Hough image corresponds to a line in the input box. \n\
      In particular, the line for hough_image[y][x] is given by get_line(point(x,y)).  \n\
      Also, when viewing the Hough image, the x-axis gives the angle of the line \n\
      and the y-axis the distance of the line from the center of the box.  The \n\
      conversion between Hough coordinates and angle and pixel distance can be \n\
      obtained by calling get_line_properties()." )
    /*!
        requires
            - box.width() == size
            - box.height() == size
        ensures
            - Computes the Hough transform of the part of img contained within box.
              In particular, we do a grayscale version of the Hough transform where any
              non-zero pixel in img is treated as a potential component of a line and
              accumulated into the returned Hough accumulator image.  However, rather than
              adding 1 to each relevant accumulator bin we add the value of the pixel
              in img to each Hough accumulator bin.  This means that, if all the
              pixels in img are 0 or 1 then this routine performs a normal Hough
              transform.  However, if some pixels have larger values then they will be
              weighted correspondingly more in the resulting Hough transform.
            - The returned hough transform image will be size rows by size columns.
            - The returned image is the Hough transform of the part of img contained in
              box.  Each point in the Hough image corresponds to a line in the input box.
              In particular, the line for hough_image[y][x] is given by get_line(point(x,y)). 
              Also, when viewing the Hough image, the x-axis gives the angle of the line
              and the y-axis the distance of the line from the center of the box.  The
              conversion between Hough coordinates and angle and pixel distance can be
              obtained by calling get_line_properties().
    !*/

        .def("__call__", &compute_ht2<uint8_t>, py::arg("img"))
        .def("__call__", &compute_ht2<uint16_t>, py::arg("img"))
        .def("__call__", &compute_ht2<uint32_t>, py::arg("img"))
        .def("__call__", &compute_ht2<uint64_t>, py::arg("img"))
        .def("__call__", &compute_ht2<int8_t>, py::arg("img"))
        .def("__call__", &compute_ht2<int16_t>, py::arg("img"))
        .def("__call__", &compute_ht2<int32_t>, py::arg("img"))
        .def("__call__", &compute_ht2<int64_t>, py::arg("img"))
        .def("__call__", &compute_ht2<float>, py::arg("img"))
        .def("__call__", &compute_ht2<double>, py::arg("img"),
            "    simply performs: return self(img, get_rect(img)).  That is, just runs the hough transform on the whole input image.")

        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<uint8_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<uint16_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<uint32_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<uint64_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<int8_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<int16_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<int32_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<int64_t>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<float>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines<double>, py::arg("img"), py::arg("box"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1,
"requires \n\
    - box.width() == size \n\
    - box.height() == size \n\
    - for all valid i: \n\
        - rectangle(0,0,size-1,size-1).contains(hough_points[i]) == true \n\
          (i.e. hough_points must contain points in the output Hough transform \n\
          space generated by this object.) \n\
    - angle_window_size >= 1 \n\
    - radius_window_size >= 1 \n\
ensures \n\
    - This function computes the Hough transform of the part of img contained \n\
      within box.  It does the same computation as __call__() defined above, \n\
      except instead of accumulating into an image we create an explicit list \n\
      of all the points in img that contributed to each line (i.e each point in \n\
      the Hough image). To do this we take a list of Hough points as input and \n\
      only record hits on these specifically identified Hough points.  A \n\
      typical use of find_pixels_voting_for_lines() is to first run the normal \n\
      Hough transform using __call__(), then find the lines you are interested \n\
      in, and then call find_pixels_voting_for_lines() to determine which \n\
      pixels in the input image belong to those lines. \n\
    - This routine returns a vector, CONSTITUENT_POINTS, with the following \n\
      properties: \n\
        - CONSTITUENT_POINTS.size == hough_points.size \n\
        - for all valid i: \n\
            - Let HP[i] = centered_rect(hough_points[i], angle_window_size, radius_window_size) \n\
            - Any point in img with a non-zero value that lies on a line \n\
              corresponding to one of the Hough points in HP[i] is added to \n\
              CONSTITUENT_POINTS[i].  Therefore, when this routine finishes, \n\
              #CONSTITUENT_POINTS[i] will contain all the points in img that \n\
              voted for the lines associated with the Hough accumulator bins in \n\
              HP[i]. \n\
            - #CONSTITUENT_POINTS[i].size == the number of points in img that \n\
              voted for any of the lines HP[i] in Hough space.  Note, however, \n\
              that if angle_window_size or radius_window_size are made so large \n\
              that HP[i] overlaps HP[j] for i!=j then the overlapping regions \n\
              of Hough space are assigned to HP[i] or HP[j] arbitrarily. \n\
              That is, we treat HP[i] and HP[j] as disjoint even if their boxes \n\
              overlap.  In this case, the overlapping region is assigned to \n\
              either HP[i] or HP[j] in an arbitrary manner." )
    /*!
        requires
            - box.width() == size
            - box.height() == size
            - for all valid i:
                - rectangle(0,0,size-1,size-1).contains(hough_points[i]) == true
                  (i.e. hough_points must contain points in the output Hough transform
                  space generated by this object.)
            - angle_window_size >= 1
            - radius_window_size >= 1
        ensures
            - This function computes the Hough transform of the part of img contained
              within box.  It does the same computation as __call__() defined above,
              except instead of accumulating into an image we create an explicit list
              of all the points in img that contributed to each line (i.e each point in
              the Hough image). To do this we take a list of Hough points as input and
              only record hits on these specifically identified Hough points.  A
              typical use of find_pixels_voting_for_lines() is to first run the normal
              Hough transform using __call__(), then find the lines you are interested
              in, and then call find_pixels_voting_for_lines() to determine which
              pixels in the input image belong to those lines.
            - This routine returns a vector, CONSTITUENT_POINTS, with the following
              properties:
                - CONSTITUENT_POINTS.size == hough_points.size
                - for all valid i:
                    - Let HP[i] = centered_rect(hough_points[i], angle_window_size, radius_window_size)
                    - Any point in img with a non-zero value that lies on a line
                      corresponding to one of the Hough points in HP[i] is added to
                      CONSTITUENT_POINTS[i].  Therefore, when this routine finishes,
                      #CONSTITUENT_POINTS[i] will contain all the points in img that
                      voted for the lines associated with the Hough accumulator bins in
                      HP[i].
                    - #CONSTITUENT_POINTS[i].size == the number of points in img that
                      voted for any of the lines HP[i] in Hough space.  Note, however,
                      that if angle_window_size or radius_window_size are made so large
                      that HP[i] overlaps HP[j] for i!=j then the overlapping regions
                      of Hough space are assigned to HP[i] or HP[j] arbitrarily.
                      That is, we treat HP[i] and HP[j] as disjoint even if their boxes
                      overlap.  In this case, the overlapping region is assigned to
                      either HP[i] or HP[j] in an arbitrary manner.
    !*/
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<uint8_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<uint16_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<uint32_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<uint64_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<int8_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<int16_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<int32_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<int64_t>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<float>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1)
        .def("find_pixels_voting_for_lines", &ht_find_pixels_voting_for_lines2<double>, py::arg("img"), py::arg("hough_points"), py::arg("angle_window_size")=1, py::arg("radius_window_size")=1,
"    performs: return find_pixels_voting_for_lines(img, get_rect(img), hough_points, angle_window_size, radius_window_size); \n\
That is, just runs the routine on the whole input image." )

        .def("find_strong_hough_points", &ht_find_strong_hough_points, py::arg("himg"), py::arg("hough_count_thresh"), py::arg("angle_nms_thresh"), py::arg("radius_nms_thresh"),
"requires \n\
    - himg has size() rows and columns. \n\
    - angle_nms_thresh >= 0 \n\
    - radius_nms_thresh >= 0 \n\
ensures \n\
    - This routine finds strong lines in a Hough transform and performs \n\
      non-maximum suppression on the detected lines.  Recall that each point in \n\
      Hough space is associated with a line. Therefore, this routine finds all \n\
      the pixels in himg (a Hough transform image) with values >= \n\
      hough_count_thresh and performs non-maximum suppression on the \n\
      identified list of pixels.  It does this by discarding lines that are \n\
      within angle_nms_thresh degrees of a stronger line or within \n\
      radius_nms_thresh distance (in terms of radius as defined by \n\
      get_line_properties()) to a stronger Hough point. \n\
    - The identified lines are returned as a list of coordinates in himg. \n\
    - The returned points are sorted so that points with larger Hough transform \n\
      values come first." 
    /*!
        requires
            - himg has size() rows and columns.
            - angle_nms_thresh >= 0
            - radius_nms_thresh >= 0
        ensures
            - This routine finds strong lines in a Hough transform and performs
              non-maximum suppression on the detected lines.  Recall that each point in
              Hough space is associated with a line. Therefore, this routine finds all
              the pixels in himg (a Hough transform image) with values >=
              hough_count_thresh and performs non-maximum suppression on the
              identified list of pixels.  It does this by discarding lines that are
              within angle_nms_thresh degrees of a stronger line or within
              radius_nms_thresh distance (in terms of radius as defined by
              get_line_properties()) to a stronger Hough point.
            - The identified lines are returned as a list of coordinates in himg.
            - The returned points are sorted so that points with larger Hough transform
              values come first.
    !*/
        );


    m.def("get_rect", [](const hough_transform& ht){ return get_rect(ht); },
        "returns a rectangle(0,0,ht.size()-1,ht.size()-1).  Therefore, it is the rectangle that bounds the Hough transform image.", 
        py::arg("ht")  );
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_transform_image (
    const numpy_image<T>& img,
    const point_transform_projective& map_point,
    long rows,
    long columns
)
{
    DLIB_CASSERT(rows > 0 && columns > 0, "The requested output image dimensions are invalid.");
    numpy_image<T> out(rows, columns);

    transform_image(img, out, interpolate_bilinear(), map_point);

    return out;
}
// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_extract_image_chip (
    const numpy_image<T>& img,
    const chip_details& chip_location 
)
{
    numpy_image<T> out;
    extract_image_chip(img, chip_location, out);
    return out;
}

template <typename T>
py::list py_extract_image_chips (
    const numpy_image<T>& img,
    const py::list& chip_locations
)
{
    dlib::array<numpy_image<T>> out;
    extract_image_chips(img, python_list_to_vector<chip_details>(chip_locations), out);
    py::list ret;
    for (const auto& i : out)
        ret.append(i);
    return ret;
}

// ----------------------------------------------------------------------------------------

template <typename T>
dpoint py_max_point(const numpy_image<T>& img)
{
    DLIB_CASSERT(img.size() != 0);
    return max_point(mat(img));
}

template <typename T>
dpoint py_max_point_interpolated(const numpy_image<T>& img)
{
    DLIB_CASSERT(img.size() != 0);
    return max_point_interpolated(mat(img));
}


// ----------------------------------------------------------------------------------------

template <typename T>
void py_zero_border_pixels (
    numpy_image<T>& img,
    long x_border_size,
    long y_border_size
)
{
    zero_border_pixels(img, x_border_size, y_border_size);
}

template <typename T>
void py_zero_border_pixels2 (
    numpy_image<T>& img,
    const rectangle& inside
)
{
    zero_border_pixels(img, inside);
}

// ----------------------------------------------------------------------------------------

template <typename T>
py::tuple py_spatially_filter_image (
    const numpy_image<T>& img,
    const numpy_image<T>& filter
)
{
    DLIB_CASSERT(filter.size() != 0);
    numpy_image<T> out;
    auto rect = spatially_filter_image(img, out, mat(filter));
    return py::make_tuple(out, rect);
}

template <typename T>
bool is_vector(
    const py::array_t<T>& m
)
{
    const size_t dims = m.ndim();
    const size_t size = m.size();
    if (dims == 1)
        return true;

    for (size_t i = 0; i < dims; ++i)
    {
        if (m.shape(i) != 1 && m.shape(i) != size)
            return false;
    }

    return true;
}

template <typename T>
py::tuple py_spatially_filter_image_separable (
    const numpy_image<T>& img,
    const py::array_t<T>& row_filter,
    const py::array_t<T>& col_filter
)
{
    DLIB_CASSERT(row_filter.size() != 0);
    DLIB_CASSERT(col_filter.size() != 0);
    DLIB_CASSERT(is_vector(row_filter), "The row filter must be either a row or column vector.");
    DLIB_CASSERT(is_vector(col_filter), "The column filter must be either a row or column vector.");

    numpy_image<T> out;
    auto rect = spatially_filter_image_separable(img, out, mat(row_filter.data(),row_filter.size()), mat(col_filter.data(),col_filter.size()));
    return py::make_tuple(out, rect);
}

// ----------------------------------------------------------------------------------------

void bind_image_classes4(py::module& m)
{

    const char* docs = "";

    register_hough_transform(m);

    m.def("transform_image", &py_transform_image<uint8_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<uint16_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<uint32_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<uint64_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int8_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int16_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int32_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<int64_t>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<float>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<double>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"));
    m.def("transform_image", &py_transform_image<rgb_pixel>, py::arg("img"), py::arg("map_point"), py::arg("rows"), py::arg("columns"),
"requires \n\
    - rows > 0 \n\
    - columns > 0 \n\
ensures \n\
    - Returns an image that is the given rows by columns in size and contains a \n\
      transformed part of img.  To do this, we interpret map_point as a mapping \n\
      from pixels in the returned image to pixels in the input img.  transform_image()  \n\
      uses this mapping and bilinear interpolation to fill the output image with an \n\
      interpolated copy of img.   \n\
    - Any locations in the output image that map to pixels outside img are set to 0." 
    /*!
        requires
            - rows > 0
            - columns > 0
        ensures
            - Returns an image that is the given rows by columns in size and contains a
              transformed part of img.  To do this, we interpret map_point as a mapping
              from pixels in the returned image to pixels in the input img.  transform_image() 
              uses this mapping and bilinear interpolation to fill the output image with an
              interpolated copy of img.  
            - Any locations in the output image that map to pixels outside img are set to 0.
    !*/
        );

    m.def("max_point", &py_max_point<uint8_t>, py::arg("img"));
    m.def("max_point", &py_max_point<uint16_t>, py::arg("img"));
    m.def("max_point", &py_max_point<uint32_t>, py::arg("img"));
    m.def("max_point", &py_max_point<uint64_t>, py::arg("img"));
    m.def("max_point", &py_max_point<int8_t>, py::arg("img"));
    m.def("max_point", &py_max_point<int16_t>, py::arg("img"));
    m.def("max_point", &py_max_point<int32_t>, py::arg("img"));
    m.def("max_point", &py_max_point<int64_t>, py::arg("img"));
    m.def("max_point", &py_max_point<float>, py::arg("img"));
    m.def("max_point", &py_max_point<double>, py::arg("img"),
"requires \n\
    - m.size > 0 \n\
ensures \n\
    - returns the location of the maximum element of the array, that is, if the \n\
      returned point is P then it will be the case that: img[P.y,P.x] == img.max()." 
    /*!
        requires
            - m.size > 0
        ensures
            - returns the location of the maximum element of the array, that is, if the
              returned point is P then it will be the case that: img[P.y,P.x] == img.max().
    !*/
        );

    m.def("max_point_interpolated", &py_max_point_interpolated<uint8_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<uint16_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<uint32_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<uint64_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<int8_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<int16_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<int32_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<int64_t>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<float>, py::arg("img"));
    m.def("max_point_interpolated", &py_max_point_interpolated<double>, py::arg("img"),
"requires \n\
    - m.size > 0 \n\
ensures \n\
    - Like max_point(), this function finds the location in m with the largest \n\
      value.  However, we additionally use some quadratic interpolation to find the \n\
      location of the maximum point with sub-pixel accuracy.  Therefore, the \n\
      returned point is equal to max_point(m) + some small sub-pixel delta." 
    /*!
        requires
            - m.size > 0
        ensures
            - Like max_point(), this function finds the location in m with the largest
              value.  However, we additionally use some quadratic interpolation to find the
              location of the maximum point with sub-pixel accuracy.  Therefore, the
              returned point is equal to max_point(m) + some small sub-pixel delta.
    !*/
        );

    m.def("zero_border_pixels", &py_zero_border_pixels<uint8_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<uint16_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<uint32_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<uint64_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<int8_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<int16_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<int32_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<int64_t>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<float>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<double>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"));
    m.def("zero_border_pixels", &py_zero_border_pixels<rgb_pixel>, py::arg("img"), py::arg("x_border_size"), py::arg("y_border_size"),
"requires \n\
    - x_border_size >= 0 \n\
    - y_border_size >= 0 \n\
ensures \n\
    - The size and shape of img isn't changed by this function. \n\
    - for all valid r such that r+y_border_size or r-y_border_size gives an invalid row \n\
        - for all valid c such that c+x_border_size or c-x_border_size gives an invalid column  \n\
            - assigns the pixel img[r][c] to 0.  \n\
              (i.e. assigns 0 to every pixel in the border of img)" 
    /*!
        requires
            - x_border_size >= 0
            - y_border_size >= 0
        ensures
            - The size and shape of img isn't changed by this function.
            - for all valid r such that r+y_border_size or r-y_border_size gives an invalid row
                - for all valid c such that c+x_border_size or c-x_border_size gives an invalid column 
                    - assigns the pixel img[r][c] to 0. 
                      (i.e. assigns 0 to every pixel in the border of img)
    !*/
        );

    m.def("zero_border_pixels", &py_zero_border_pixels2<uint8_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<uint16_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<uint32_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<uint64_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<int8_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<int16_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<int32_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<int64_t>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<float>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<double>, py::arg("img"), py::arg("inside"));
    m.def("zero_border_pixels", &py_zero_border_pixels2<rgb_pixel>, py::arg("img"), py::arg("inside"),
"ensures \n\
    - The size and shape of img isn't changed by this function. \n\
    - All the pixels in img that are not contained inside the inside rectangle \n\
      given to this function are set to 0.  That is, anything not \"inside\" is on \n\
      the border and set to 0." 
    /*!
        ensures
            - The size and shape of img isn't changed by this function.
            - All the pixels in img that are not contained inside the inside rectangle
              given to this function are set to 0.  That is, anything not "inside" is on
              the border and set to 0.
    !*/
        );



    m.def("spatially_filter_image", &py_spatially_filter_image<uint8_t>, py::arg("img"), py::arg("filter"));
    m.def("spatially_filter_image", &py_spatially_filter_image<float>,   py::arg("img"), py::arg("filter"));
    m.def("spatially_filter_image", &py_spatially_filter_image<double>,  py::arg("img"), py::arg("filter"),
"requires \n\
    - filter.size != 0 \n\
ensures \n\
    - Applies the given spatial filter to img and returns the result (i.e. we  \n\
      cross-correlate img with filter).  We also return a rectangle which \n\
      indicates what pixels in the returned image are considered non-border pixels \n\
      and therefore contain output from the filter.  E.g. \n\
        - filtered_img,rect = spatially_filter_image(img, filter) \n\
      would give you the filtered image and the rectangle in question.  Since the \n\
      returned image has the same shape as img we fill the border pixels by setting \n\
      them to 0. \n\
 \n\
    - The filter is applied such that it's centered over the pixel it writes its \n\
      output into.  For centering purposes, we consider the center element of the \n\
      filter to be filter[filter.shape[0]/2,filter.shape[1]/2].  This means that \n\
      the filter that writes its output to a pixel at location point(c,r) and is W \n\
      by H (width by height) pixels in size operates on exactly the pixels in the \n\
      rectangle centered_rect(point(c,r),W,H) within img." 
    /*!
        requires
            - filter.size != 0
        ensures
            - Applies the given spatial filter to img and returns the result (i.e. we 
              cross-correlate img with filter).  We also return a rectangle which
              indicates what pixels in the returned image are considered non-border pixels
              and therefore contain output from the filter.  E.g.
                - filtered_img,rect = spatially_filter_image(img, filter)
              would give you the filtered image and the rectangle in question.  Since the
              returned image has the same shape as img we fill the border pixels by setting
              them to 0.

            - The filter is applied such that it's centered over the pixel it writes its
              output into.  For centering purposes, we consider the center element of the
              filter to be filter[filter.shape[0]/2,filter.shape[1]/2].  This means that
              the filter that writes its output to a pixel at location point(c,r) and is W
              by H (width by height) pixels in size operates on exactly the pixels in the
              rectangle centered_rect(point(c,r),W,H) within img.
    !*/
        );

    m.def("spatially_filter_image_separable", &py_spatially_filter_image_separable<uint8_t>, py::arg("img"), py::arg("row_filter"), py::arg("col_filter"));
    m.def("spatially_filter_image_separable", &py_spatially_filter_image_separable<float>,   py::arg("img"), py::arg("row_filter"), py::arg("col_filter"));
    m.def("spatially_filter_image_separable", &py_spatially_filter_image_separable<double>,  py::arg("img"), py::arg("row_filter"), py::arg("col_filter"),
"requires \n\
    - row_filter.size != 0 \n\
    - col_filter.size != 0 \n\
    - row_filter and col_filter are both either row or column vectors.  \n\
ensures \n\
    - Applies the given separable spatial filter to img and returns the result \n\
      (i.e. we cross-correlate img with the filters).  In particular, calling this \n\
      function has the same effect as calling the regular spatially_filter_image() \n\
      routine with a filter, FILT, defined as follows:  \n\
        - FILT(r,c) == col_filter(r)*row_filter(c) \n\
      Therefore, the return value of this routine is the same as if it were \n\
      implemented as:    \n\
        return spatially_filter_image(img, FILT) \n\
      Except that this version should be faster for separable filters." 
    /*!
        requires
            - row_filter.size != 0
            - col_filter.size != 0
            - row_filter and col_filter are both either row or column vectors. 
        ensures
            - Applies the given separable spatial filter to img and returns the result
              (i.e. we cross-correlate img with the filters).  In particular, calling this
              function has the same effect as calling the regular spatially_filter_image()
              routine with a filter, FILT, defined as follows: 
                - FILT(r,c) == col_filter(r)*row_filter(c)
              Therefore, the return value of this routine is the same as if it were
              implemented as:   
                return spatially_filter_image(img, FILT)
              Except that this version should be faster for separable filters.
    !*/
        );
}


