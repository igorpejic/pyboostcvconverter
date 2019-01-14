#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/core.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudaarithm.hpp"

namespace pbcvt {

    using namespace boost::python;

/**
 * @brief Example function. Basic inner matrix product using explicit matrix conversion.
 * @param left left-hand matrix operand (NdArray required)
 * @param right right-hand matrix operand (NdArray required)
 * @return an NdArray representing the dot-product of the left and right operands
 */
    PyObject *dot(PyObject *left, PyObject *right) {

        Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
        pMOG2 = cv::createBackgroundSubtractorMOG2(); //MOG2 approach
        cv::Mat leftMat, rightMat;
        leftMat = pbcvt::fromNDArrayToMat(left);
        rightMat = pbcvt::fromNDArrayToMat(right);
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        // Check that the 2-D matrices can be legally multiplied.
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;
        PyObject *ret = pbcvt::fromMatToNDArray(result);
        return ret;
    }

    boost::python::tuple apply(cv::Mat frame) {
       static Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
       cv::Mat fgMask;
       cv::Mat bgImg;       static int initialized = 0;
       if (initialized == 0) {
           pMOG2 = cv::createBackgroundSubtractorMOG2(); //MOG2 approach
       }
       if (initialized == 0) {
           initialized = 1;
       }
       //_, foreground = cv2.threshold(foreground, 254, 1, cv2.THRESH_BINARY)       pMOG2->apply(frame, fgMask);
       cv::threshold(fgMask, fgMask, 254, 1, cv::THRESH_BINARY);       // foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self._KERNEL)
       // _KERNEL = numpy.ones((3, 3), numpy.uint8)
       int kdata[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
       Mat kernel(3, 3, CV_8U, kdata);
       //cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
       pMOG2->getBackgroundImage(bgImg);
       return boost::python::make_tuple(fgMask, bgImg);
    }

   boost::python::tuple applyGPU(cv::Mat frame) {
       static Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
       cv::cuda::GpuMat d_fgmask;
       cv::cuda::GpuMat d_bgimg;
       cv::cuda::GpuMat d_frame;

       //_, foreground = cv2.threshold(foreground, 254, 1, cv2.THRESH_BINARY)
       Mat fgmask;
       Mat fgimg;
       Mat bgimg;

       d_frame.upload(frame);

       static int initialized = 0;
       if (initialized == 0) {
           pMOG2 = cv::cuda::createBackgroundSubtractorMOG2(); //MOG2 approach
       }

       if (initialized == 0) {
           initialized = 1;
       }
       pMOG2->apply(d_frame, d_fgmask);
       pMOG2->getBackgroundImage(d_bgimg);
       //cv::cuda::threshold(d_fgmask, d_fgmask, 254, 1, cv::THRESH_BINARY);       // foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self._KERNEL)

       d_fgmask.download(fgmask);
       d_frame.download(frame);
       return boost::python::make_tuple(fgmask, frame);
   }

/**
 * @brief Example function. Simply makes a new CV_16UC3 matrix and returns it as a numpy array.
 * @return The resulting numpy array.
 */

	PyObject* makeCV_16UC3Matrix(){
		cv::Mat image = cv::Mat::zeros(240,320, CV_16UC3);
		PyObject* py_image = pbcvt::fromMatToNDArray(image);
		return py_image;
	}

//
/**
 * @brief Example function. Basic inner matrix product using implicit matrix conversion.
 * @details This example uses Mat directly, but we won't need to worry about the conversion in the body of the function.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
    cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;

        return result;
    }

    /**
     * \brief Example function. Increments all elements of the given matrix by one.
     * @details This example uses Mat directly, but we won't need to worry about the conversion anywhere at all,
     * it is handled automatically by boost.
     * \param matrix (numpy array) to increment
     * \return
     */
    cv::Mat increment_elements_by_one(cv::Mat matrix){
        matrix += 1.0;
        return matrix;
    }


#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
        matFromNDArrayBoostConverter();

        //expose module-level functions
        def("dot", dot);
        def("dot2", dot2);
		def("apply", apply);
		def("applyGPU", applyGPU);
		def("makeCV_16UC3Matrix", makeCV_16UC3Matrix);

		//from PEP8 (https://www.python.org/dev/peps/pep-0008/?#prescriptive-naming-conventions)
        //"Function names should be lowercase, with words separated by underscores as necessary to improve readability."
        def("increment_elements_by_one", increment_elements_by_one);
    }

} //end namespace pbcvt
