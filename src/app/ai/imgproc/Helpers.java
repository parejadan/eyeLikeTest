package app.ai.imgproc;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.CvType;
import org.opencv.core.MatOfDouble;

public class Helpers {

	//private static final String helpTAG = "imgproc::Helpers";
	
	public Helpers() {};
	
	public boolean rectInImage(Rect rect, Mat image) {
		return rect.x > 0 && rect.y > 0 && rect.x + rect.width < image.cols() &&
				rect.y + rect.height < image.rows();
	}
	
	public boolean inMat(Point p, int rows, int cols) {
		return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
	}
	
	public Mat matrixMagnitude(Mat matX, Mat matY, int rows, int cols) {
		Mat mags = Mat.zeros(rows, cols, CvType.CV_64F);
		/*for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				mags.put(y, x,
						Math.sqrt(
								Math.pow(matX.get(y, x)[0], 2.0) + Math.pow(matY.get(y, x)[0], 2.0)
										));
			}
		}*/
		//compute the magnitude of each pixel
		Core.add(matX.mul(matX), matY.mul(matY), mags);
		Core.sqrt(mags, mags);
		
		return mags;
	}

	public Mat matrixMagnitude(Mat matX, Mat matY) {
		int rows = matX.rows()-1, cols = matX.cols()-1;
		Mat mags = Mat.zeros(rows, cols, CvType.CV_64F);
		/*for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				mags.put(y, x,
						Math.sqrt(
								Math.pow(matX.get(y, x)[0], 2.0) + Math.pow(matY.get(y, x)[0], 2.0)
										));
			}
		}*/
		//compute the magnitude of each pixel
		Core.add(matX.mul(matX), matY.mul(matY), mags);
		Core.sqrt(mags, mags);
		
		return mags;
	}

	
	public double computeDynamicThreshold(Mat mat, double stdDevFactor) {
		MatOfDouble std = new MatOfDouble(), mean = new MatOfDouble();
		Core.meanStdDev(mat, mean, std);
		double stdDev = std.get(0,0)[0] / Math.sqrt(mat.rows()*mat.cols());
		return stdDevFactor *stdDev + mean.get(0,0)[0];
	}
}
