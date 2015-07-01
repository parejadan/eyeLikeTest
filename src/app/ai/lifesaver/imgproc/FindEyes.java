/**
 * This file contains the code used for eye detection. Face must already 
 *  be located and passed to detect() parameter
 */

package app.ai.lifesaver.imgproc;

import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Mat;
//import org.opencv.core.Scalar;
import org.opencv.core.Size;
//import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

public class FindEyes {
	public static int eyeRegW, eyeRegH;
	public static double[] cEye;
	private static int rows, cols;
	
	private Helpers hp = new Helpers();
	
	public FindEyes() {
		cEye = new double[2];
	}
		
	public Point unscalePoint(Point p, Rect origSize) {
		float ratio = ( (float) Vars.kFastEyeWidth/origSize.width );
		int x = (int) (p.x / ratio);
		int y = (int) (p.y / ratio);
		return (new Point(x, y) );
	}
	
	public  Mat scaleToFastSize(Mat src) {
		Mat resized = src.clone();
		if (resized.cols() > 0)
		Imgproc.resize(src, resized, new Size(Vars.kFastEyeWidth, Vars.kFastEyeWidth/src.cols() * src.rows()));
			
		return resized;
	}

	void testPossibleCenterFormula(int x, int y, Mat weight, double gx, double gy, Mat out) {
		// Precompute values
		int xy_sum = x*x + y*y, y_double = 2*y, x_double = 2*x;
		double gxy = gx*x + gy*y, dot, top;
		double[] bufO;
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				//avoid a specific location
				if (j == y && k == x) continue;

				// see if the result will be negative
				top = gxy - gx*k - gy*j;
				if (top <= 0) dot = 0; 
				else dot = top / Math.sqrt(xy_sum + k*(k - x_double) + j*(j - y_double));

				bufO = out.get(j, k);
				if (Vars.kEnableWeight) bufO[0] += dot*dot *(weight.get(j, k)[0]/Vars.kWeightDivisor);
				else bufO[0] += dot*dot;
				
				out.put(j, k, bufO);	
			}
		}
	}
	
	Mat computeGradient(Mat mat, int rows, int cols) {
		Mat out = Mat.zeros(rows, cols, CvType.CV_64F);
		
		for (int y = 0; y < rows; y++) {
			out.put( y, 0, mat.get(y, 1)[0] - mat.get(y, 0)[0] );
			
			for (int x = 1; x < cols - 1; x++)
				out.put( y, x, (mat.get(y, x+1)[0] - mat.get(y, x-1)[0]) / 2.0 );
			
			out.put(y, cols-1, mat.get(y, cols-1)[0] - mat.get(y, cols-2)[0]);
		}
		return out;
	}

	public Point findEyeCenter(Mat face, Rect eye) {
		double gx, gy, gradientThresh;
		Mat eyeROI = scaleToFastSize( face.submat(eye) );
		rows = eyeROI.rows()-1; cols = eyeROI.cols()-1;
		
		Core.rectangle(face, eye.tl(), eye.br(), Vars.FACE_RECT_COLOR); //draw eye region - for debugging
    	//find the gradient
		Mat gradX = computeGradient(eyeROI, rows, cols);
		Mat gradY = computeGradient(eyeROI.t(), cols, rows).t();
		//compute all pixel magnitudes
		Mat mags = hp.matrixMagnitude(gradX, gradY, rows, cols);
		gradientThresh = hp.computeDynamicThreshold(mags, Vars.kGradientThresh); //compute all pixel threshold
		//normalize
		for (int y = 0; y < rows; y++)
			for (int x = 0; x < cols; x++) {
				if (mags.get(y, x)[0] > gradientThresh) {
					gradX.put(y, x, gradX.get(y, x)[0]/mags.get(y, x)[0]);
					gradY.put(y, x, gradY.get(y, x)[0]/mags.get(y, x)[0]);
				} else {
					gradX.put(y, x, 0.0);
					gradY.put(y, x, 0.0);
				}
			}
		//create a blurred and inverted image for weighting
		Mat weight = new Mat();
		Imgproc.GaussianBlur(eyeROI, weight, (new Size(Vars.kWeightBlurSize, Vars.kWeightBlurSize)), 0, 0);
		for (int y = 0; y < rows+1; y++)
			for (int x = 0; x < cols+1; x++) {
				double[] tmp =  weight.get(y, x);
				tmp[0] = 255 - tmp[0];
				weight.put(y, x, tmp);
			}
		//run algorithm
		Mat outSum = Mat.zeros(rows, cols, CvType.CV_64F);
		for (int y = 0; y < rows; y++)
			for (int x = 0; x <  cols; x++) {
				gx = gradX.get(y, x)[0]; gy = gradY.get(y, x)[0];
				if (gx == 0.0 && gy == 0.0) {
					continue;
				}
				testPossibleCenterFormula(x, y, weight, gx, gy, outSum);
			}
		//find the maximum point
		Core.MinMaxLocResult mmr = Core.minMaxLoc(outSum);
		return unscalePoint(mmr.maxLoc, eye);
		//Point e = unscalePoint(mmr.maxLoc, eye);
		
		//outSum.release(); weight.release(); gradX.release(); gradY.release(); mags.release();
		//outSum = weight = gradX = gradY = mags = null;
		//return e;
	}
		
	public Point findPupil(Mat frame_gray, Rect face, boolean leftEye) {
		Mat faceROI = frame_gray.submat(face);
		Point pupil = null;
		//faceOLD = frame.clone();
		/*if (Vars.kSmoothFaceImage)
			Imgproc.GaussianBlur(faceROI, faceROI, (new Size(0, 0)), Vars.kSmoothFaceFactor * face.width);*/

		//determine a face to eye ratio and positioning based on percentages
		eyeRegW = (int) (face.width * (Vars.kEyePercentWidth/100.0));
		eyeRegH = (int) (face.height * (Vars.kEyePercentHeight/100.0));
		int eyeDexY = (int) (face.height * (Vars.kEyePercentTop/100.0));
		int eyeDexX = (int) (face.width * (Vars.kEyePercentSide/100.0));
		Rect lEyeReg = new Rect(eyeDexX, eyeDexY, eyeRegW, eyeRegH);
		double[] cFace = {face.width/2.0 + face.x, face.height/2.0 + face.y};

		if (leftEye) { //detect pupils for left eye region (the camera's left, not the model's)
			// 4 and 1 accommodate minor int arithmetic value lost in calculations
			cEye[0] = cFace[0] - lEyeReg.width - 4; cEye[1] = cFace[1] - lEyeReg.y - 1;
			pupil = findEyeCenter(faceROI,lEyeReg);
			pupil.x += cEye[0];
			pupil.y += cEye[1];

		} else {
			Rect rEyeReg = new Rect(face.width - eyeRegW - eyeDexX, eyeDexY, eyeRegW,eyeRegH);
			cEye[0] = cFace[0] + 4; cEye[1] = cFace[1] - rEyeReg.y - 1;
			pupil = findEyeCenter(faceROI,rEyeReg);
			pupil.x += cEye[0];
			pupil.y += cEye[1];
		}
		//Core.circle(frame_gray, new Point(cEye[0], cEye[1]), 2, Vars.EYE_CIRCLE_COLR);
		//Core.circle(frame_gray, new Point(eyeDexX+face.x, eyeDexY+face.y), 2, Vars.EYE_CIRCLE_COLR);
		//Core.circle(frame_gray, new Point(face.x, face.y), 2, Vars.EYE_CIRCLE_COLR);
		//System.out.println(1.0*(cEye[1]-face.y)/face.height);

		faceROI.release(); faceROI = null;
		return pupil;
	}

///////////////////////////////////////////ASCENT CODE UNDER CONSTRUCTION	
/*
	public Point findEyeCenter(Mat face, Rect eye) {
		//double gx, gy, gradientThresh;
		double gradientThresh;
		Mat eyeROI = scaleToFastSize( face.submat(eye) );
		Core.rectangle(face, eye.tl(), eye.br(), Vars.FACE_RECT_COLOR); //draw eye region - for debugging
    	//find the gradient
		Mat gradX = computeGradient(eyeROI);
		Mat gradY = computeGradient(eyeROI.t()).t();
		//compute all pixel magnitudes
		Mat mags = hp.matrixMagnitude(gradX, gradY);
		gradientThresh = hp.computeDynamicThreshold(mags, Vars.kGradientThresh); //compute all pixel threshold
		//normalize
		int rows  = eyeROI.rows()-1, cols = eyeROI.cols()-1;
		for (int y = 0; y < rows; y++)
			for (int x = 0; x < cols; x++) {
				if (mags.get(y, x)[0] > gradientThresh) {
					gradX.put(y, x, gradX.get(y, x)[0]/mags.get(y, x)[0]);
					gradY.put(y, x, gradY.get(y, x)[0]/mags.get(y, x)[0]);
				} else {
					gradX.put(y, x, 0.0);
					gradY.put(y, x, 0.0);
				}
			}
		//create a blurred and inverted image for weightingS
		Point e = centerLocalisation(gradX, gradY);
		gradX.release(); gradY.release(); mags.release(); eyeROI.release();
		gradX = gradY = mags = eyeROI = null;

		return e;
	}
*/
	Mat computeGradient(Mat mat) {
		int rows = mat.rows()-1, cols = mat.cols()-1;
		Mat out = Mat.zeros(rows, cols, CvType.CV_64F);
		
		for (int y = 0; y < rows; y++) {
			out.put( y, 0, mat.get(y, 1)[0] - mat.get(y, 0)[0] );
			
			for (int x = 1; x < cols - 1; x++)
				out.put( y, x, (mat.get(y, x+1)[0] - mat.get(y, x-1)[0]) / 2.0 );
			
			out.put(y, cols-1, mat.get(y, cols-1)[0] - mat.get(y, cols-2)[0]);
		}
		return out;
	}

	/** Perform gradient ascent on the normalized gradients g. (CITE)
	 * Pre-conditions: gradX and gradY are normalized.
	 * @return the best center found wrt the objective function
	 */
	Point centerLocalisation(Mat gradX, Mat gradY) {
		int rows = gradX.rows()-1, cols = gradX.cols()-1;
		// Maximum number of iterations
		int t_max = (int) Math.ceil(Math.sqrt(rows*cols));
		// starting step size
		int start_step = 2;
		int x, y, j;
		// Threshold to check convergence
		double threshold = .01;
		double max_objective = -100000000000.0; 
		Point best, current, old;
		// x:=step size; y:=new objective value
		Point computeInfo = new Point(start_step, 0);
		// find a single point's derivative gradient
		Point g_pt;
		//Point tmp = new Point();
		// initialize placeholder best
		//System.out.printf("center localisation\n");
		best = new Point();
		for(x=0; x<cols; x++)
			for(y=0; y<rows; y++)
				if (gradX.get(y,x)[0] != 0) { // ensure a significant magnitude
					current = new Point(x,y);
					computeInfo.x = start_step;
					//System.out.printf("c.y: %f\t|c.x: %f\n",current.y,current.x);
					for(j=0; j<t_max; j++) {
						old = current;
						//System.out.printf("%d)\tc.y: %f\t|c.x: %f\n",j, current.y,current.x);
						g_pt = computeGradientPoint(current, gradX, gradY);
						
						// use Armijo's rule - does the point return correctly? How can points be returned
						computeInfo = computeStepSize(current, gradX, gradY, g_pt, computeInfo.x, threshold);
						//System.out.println(computeInfo.x);
						//System.out.println(j);
						current.x += computeInfo.x * g_pt.x;
						current.y += computeInfo.x * g_pt.y;
						// Check if image borders have been reached
						/*faceTMP = faceOLD.clone();
						tmp.x = current.x + CCCC[0];
						tmp.y = current.y + CCCC[1];
						Core.circle(faceTMP,  tmp,  3,  new Scalar(0, 255, 0));
						tmp.x = old.x + CCCC[0];
						tmp.y = old.y + CCCC[1];
						Core.circle(faceTMP,  old, 5,  new Scalar(0, 0, 255));
						tmp.x = best.x + CCCC[0];
						tmp.y = best.y + CCCC[1];
						Core.circle(faceTMP,  best,  7,  new Scalar(255, 0, 0));
			        	Highgui.imwrite("".format("%d-%d-detection.png",j, y), faceTMP);*/
						
						if (current.x > cols || current.x < 0 ||
							current.y > rows || current.y < 0) {
							/*System.out.println("break check");*/ break;
						}
						
						// if euclidean distance is less than the threshold, converged
						if (length(current, old) <= threshold) {
							// update maximum value, if necessary
							if (computeInfo.y > max_objective) {
								max_objective = computeInfo.y;
								best = current.clone();
							}
							break; // stop; converged
						}
					}
				}
		if (best.x == 0 && best.y ==0)
			System.out.println("welp, shit");
		//Core.circle(faceROI,  pupil,  5,  Vars.EYE_CIRCLE_COLR);

		return best;
	}

	/** Computes the gradient with respect to the center point c using gradient ascent. -see EQ 2.3
	 * @param gradX - the pixel gradients wrt the x-direction.
	 * @param gradY - the pixel gradients wrt the y-direction.
	 * @return a point where x-value = derivative wrt c_x, & x-value = derivative wrt c_y
	 */
	Point computeGradientPoint(Point c, Mat gradX, Mat gradY) {
		int rows = gradX.rows()-1, cols = gradX.cols()-1;
		// variables related to k
		double xik, gik, ck;
		// intermediate variables
		double e_i, n_i_sq;
		// summation variable
		double sum;
		Point grad = new Point();

		int k, x, y;
		for (k=1; k<=2; k++) {
			sum = 0;
			if (k==1) ck = c.x;
			else ck = c.y;

			for(x=0; x<cols; x++) {
				xik = x;
				for(y=0; y<rows; y++) {
					if (c.y == y && c.x == x) continue;
					if (k==1) // get x-gradient of g_i
						gik = gradX.get(y, x)[0];
					else {
						xik = y;
						gik = gradY.get(y, x)[0]; // get x-gradient of g_i
					}

					// dot product of g_i and (x_i - c)T
					e_i = (x - c.x) * gradX.get(y, x)[0] + (y - c.y) * gradY.get(y, x)[0];
					// length between x_i and c
				    n_i_sq = length(new Point(x, y), c);
				    n_i_sq *= n_i_sq;
					sum += (e_i * (((xik - ck) * e_i) - (gik * n_i_sq))) / (n_i_sq * n_i_sq);
					//if (n_i_sq ==  0)
						//System.out.println(n_i_sq);
					//System.out.printf("%d)\tcx: %f\t|cy: %f\t|gik: %f\t|sum: %f\n", y, c.x, c.y, gik, sum);
				}
			}
			
			// Set the corresponding gradient
			if (k==1) grad.x = 2*sum/((rows + 1)*(cols + 1));
			else grad.y = 2*sum/((rows + 1)*(cols + 1));
		}

		return grad;
	}

	/** Computes the value of the objective function wrt a center c.
	 * @param gradX - pixel gradients wrt the x-direction.
	 * @param gradY - pixel gradients wrt the y-direction.
	 * @return the objective value for a center c.
	 */
	double computeObjective(Point c, Mat gradX, Mat gradY) {
		int rows = gradX.rows()-1, cols = gradX.cols()-1;
		double objective = 0, d_i;

		int x, y;
		double tmpY, tmpX, tmpL;
		//System.out.printf("c.y: %f\t|c.x: %f\n",c.y,c.x);
		for(x=0; x<cols; x++)
			for(y=0; y<rows; y++) {
				if (c.y == y && c.x == x) continue;
				
				tmpY = gradY.get(y,x)[0];
				tmpX = gradX.get(y,x)[0];
				tmpL = length(new Point(x,y), c);
				// dot product: d_iT*g_i
				//System.out.printf("gY: %f\t|gX: %f|l: %f\n",tmpY, tmpX, tmpL);
				d_i = ((y - c.y)*tmpY + (x - c.x)*tmpX) / tmpL;

				objective += d_i*d_i;
			}

		return objective/((rows+1)*(cols+1));
	}

	/** Calculates step size using Armijo's rule
	 * @return Point where x-value = the step size, & y-value = the new objective value
	 */
	Point computeStepSize(Point center, Mat gradX, Mat gradY, Point deriv, double prevSize, double threshold) {
		//System.out.printf("c.y: %f\t|c.x: %f\n",center.y,center.x);

		Point grad = new Point(gradX.get((int)center.y, (int)center.x)[0], gradY.get((int)center.y, (int)center.x)[0]);
		Point cent = center.clone(); // protect center from being changed
		//System.out.println(prevSize);
		double size = prevSize * 1.5; // this may or may not be good !!!
		double c = 10e-4; // as recommended by Nocedal
		// c * p_kT * part_deriv(f(center))
		double dot = c * (grad.x*deriv.x + grad.y*deriv.y);
		double current_objective = computeObjective(center, gradX, gradY);

		double right = current_objective + size * dot;
		//System.out.printf("co: %f\t|s: %f\t|d: %f\t|cx: %f\t|cy: %f\n",current_objective, size, dot,center.x, center.y);

		cent.x += size * grad.x;
		cent.y += size * grad.y;
		double left = computeObjective(cent, gradX, gradY);

		// while armijo's rule is not fulfilled, reduce step size
		//int cnt = 0;
		//System.out.printf("l: %f\t|r: %f\t|t: %f\t|s: %f\n",left, right, threshold, size);
		while (left < right && threshold < size) {
			// undo the previous step
			cent.x -= size * grad.x;
			cent.y -= size * grad.y;
			// step smaller
			size *= .5;
			right = current_objective + size * dot;
			cent.x += size * grad.x;
			cent.y += size * grad.y;
			left = computeObjective(cent, gradX, gradY);
			//cnt++;
		}
		//System.out.printf("while-loop iterations within stepsize: %d\n",cnt);
		//System.out.printf("l: %f\t|r: %f\t|t: %f\t|s: %f\t|j: ",left, right, threshold, size);

		// package information to save CPU cycles
		Point information = new Point(size, left);
		return information;
	}

	// Computes the distance between points a and b.
	double length(Point a, Point b) { return Math.sqrt((a.x - b.x)*(a.x - b.x) + (a.y + b.y)*(a.y + b.y)); }	

///////////////////////////////////////////ASCENT CODE UNDER CONSTRUCTION
	
}
