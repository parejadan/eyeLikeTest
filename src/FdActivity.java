import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Date;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.objdetect.CascadeClassifier;

import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import app.ai.imgproc.FindEyes;
import app.ai.imgproc.FindFace;
import app.ai.imgproc.ProcStruct;
import app.ai.stats.Stats;
import app.ai.core.CoreVars;
import app.ai.core.LimQueue;

import java.util.Scanner;

public class FdActivity {

	public static void main(String[] args) throws Exception {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);    	
        Detect obj = new Detect();
       
        while (true)
        	obj.nextFrame();
       // String inD = "input", t2 = "d-mul", outD = "output", vt = "mp4", dt = "arff";  
       // obj.adjust = true;
       // obj.captureFromVidFile(30, 4, 3.5, 0, String.format("%s/s1/%s-s1.%s", inD, t2, vt), String.format("%s/%s-s4.%s", outD, t2, dt));

    }
}

class Detect {
	static String trainFile = "merged.arff";
    //APIs
    private static FindFace _faceD; //face detection object
    private static FindEyes _eyeD; //pupil detection object
    private static Stats _stool = new Stats(); //mathematics and stats object
    public static LimQueue<Double> mags;
    private static VideoCapture cam;
    FastVector fvWekaAttributes;
    static String[] labels = {"not-distracted", "distracted"}; // or non-distracted
    Scanner input = new Scanner(System.in), fi_d, fi_nd;
	//PUPIL DETECTION OBJECTS
    Point last, cur; 
    ProcStruct ps;
	Mat frame  = new Mat();
	boolean adjust;
	int frms = 0, type;
		
	Detect() {
		_faceD = new FindFace( new CascadeClassifier(getClass().getResource("/haarcascade_frontalface_alt.xml").getPath()) );
		_eyeD = new FindEyes();
		_stool = new Stats();
		 cam = new VideoCapture(0);
		 mags = new LimQueue<Double>(1);
	}

    /**
     * Calculates a video's frame rate by counting overall frames and  devide them
     * by the specified video length vidLen. vidFile is string name of video to get frame count from.
     */
	public double getFPS(String vidFile, double vidLen) {
		cam = new VideoCapture(vidFile);
		System.out.println("obtaining video fps");
		do {
			frms++;
			cam.read(frame);
		} while ( !frame.empty() );
		System.out.printf("recorded frames: %d\t|reported time: %f s\t|calculated FPS: %f\n", frms, vidLen, frms / vidLen);
		return frms / vidLen;
	}
	
	/**
	 * Reads in frames and draws detections for demonstration purposes
	 */
	void nextFrame() {
		cam.read(frame);
		if (adjust) {
			Core.transpose(frame, frame);
			Core.flip(frame,  frame, 1);
		}
		ps = _faceD.getFace(frame);

        if (ps.getRect() == null) { //face cannot be located
			ps.setpupil( new org.opencv.core.Point(-1, 1) );
		} else {
			_eyeD.getPupil(ps.getImg(), ps.getRect(), true);
		}
    	Highgui.imwrite( "detection.png", ps.getImg() );
	}
	
    /**
     * Reads in the next frame from a camera object. Pupil coordinates from each frame read is extracted,
     * then stored for calculating distance between two consecutive frames. Magnitudes are then stored in a global LimQueue object.
     */
    boolean readNextFrame() {
		cam.read(frame); //read in next frame
		
		//adjusts video orientation if recorded veritcally instead of horizontally 
		if (adjust) {
			Core.transpose(frame, frame);
			Core.flip(frame, frame, 1);
		}
		if ( frame.empty() ) {
			mags.add(0.0);
			return false; //no more video feed
		}
		
    	ps = _faceD.getFace(frame);
        if (ps.getRect() == null) { //face cannot be located
            ps.setpupil( new org.opencv.core.Point(-1, -1) );
        } else { //locate right pupil within face region
            ps.setpupil(_eyeD.getPupil(ps.getImg(), ps.getRect(), true));
        }
    	cur = ps.getpupil();
    	
        //treat closed eye signals as no eye movement
        if (cur.x  ==  -1 && cur.y == -1) {
        	mags.add(0.0);
            last = cur; //rotate coordinate for next computation
            return true; //there was video feed, but no pupil to extract
        } else if (last.x == -1) {
        	mags.add( _stool.length(0.0, 0.0, cur.x, cur.y) );
        } else if (cur.x == -1) {
        	mags.add( _stool.length(last.x, last.y, 0.0, 0.0) );
        } else {
        	mags.add( _stool.length(last.x, last.y, cur.x, cur.y) );
        }
        last = cur; //rotate coordinate for next computation
    	frame.release();
    	return true;
    }
      
    /**
     * Resets global LimQueue and fills it with magnitudes of the specified length
     * return: true when queue is full, false otherwise
     */
    boolean fillUpQueue() {
    	mags = new LimQueue<Double>(CoreVars.queueSize); //reset mags
        
        //FILL UP THE MAGNITUDE QUEUE
		cam.read(frame); //read in an initial frame
		//adjusts video orientation if recorded vertically instead of horizontally 
		if (adjust) {
			Core.transpose(frame, frame);
			Core.flip(frame, frame, 1);
		}
		
    	ps = _faceD.getFace(frame);
        if (ps.getRect() == null) { //face cannot be located
        	System.out.println("Face not found");
            ps.setpupil( new org.opencv.core.Point(-1, -1) );
        } else { //locate right pupil within face region
        	System.out.println("Face found");
            ps.setpupil(_eyeD.getPupil(ps.getImg(), ps.getRect(), true));
        }
        last = ps.getpupil();
        
        while ( mags.hasRoom() ) readNextFrame(); //continue reading frames if not full yet
        
    	return !mags.hasRoom();
    }
    
    /**
     * Tests prediction accuracy on a given video
     * 
     * @param vidFPS - video's frames per second. If unknown, pass getFPS()
     * @param startT - in seconds, time stamp in video where to start testing training data
     * @param stampT - same value as that passed training data
     * @param vidLenSecs - video length in seconds
     * @param vidFile - string to of video file to test training data on, should include extension and path
     * @param trainData - training data to test
     */
    void testOnVideo(double vidFPS, int startT, int stampT, int vidLenSecs, String vidFile, String trainData) throws Exception {
    	System.out.println("Start video offline testing..");
    	System.out.println("Seting up image processing..");    	
    	//SETUP IMAGE PROCESSING
    	trainFile = trainData; //data to train classifier on for making predictions
    	cam = new VideoCapture(vidFile);
    	CoreVars.queueSize = (int) (stampT*vidFPS);
    	CoreVars.rows = (int) (vidLenSecs*vidFPS);
    	
    	System.out.println("Setting up prediction objects..");
    	//SETUP WEKA CLASSIFIER
    	Attribute[] attributes = new Attribute[CoreVars.queueSize+1];
        fvWekaAttributes = new FastVector(4); // Declare the feature vector
    	for (int i = 0; i < CoreVars.queueSize; i++) {
    		attributes[i] = new Attribute(String.format("magnitude%d", i) ); // create numeric attributes
    		fvWekaAttributes.addElement(attributes[i]);
    	}
        FastVector fvClassVal = new FastVector(2); // Declare the class attribute along with its values
        fvClassVal.addElement(labels[0]);
        fvClassVal.addElement(labels[1]);
        attributes[CoreVars.queueSize] = new Attribute("theClass", fvClassVal);
        fvWekaAttributes.addElement(attributes[CoreVars.queueSize]);
    	System.out.println("--Loading dataset for training classifier..");
    	Instances isTrainingSet = loadArff(trainFile);
    	isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);
    	// After training data set is complete, build model
        Classifier cModel = (Classifier)new RandomForest();   
        cModel.buildClassifier(isTrainingSet);
        Evaluation eTest = new Evaluation(isTrainingSet);
        Instances dataUnlabeled = new Instances("TestInstances", fvWekaAttributes, 0);
        
        System.out.println("Fast forwarding video to desired time stamp..");
    	for (int i = 0; i < (int) (startT*vidFPS); i++) cam.read(frame); //fast forward video
        System.out.println("Reached desired time stamp");
    	System.out.println("Setting up feedback system..");
        double isDistracted = 0;
        
        System.out.println("Filling up queue for monitoring");
        FeedBack feed = new FeedBack();
        Instance newFrame;
        System.out.println("Queue filled up..");
        fillUpQueue();
        System.out.printf( "Ready to do frame by frame monitoring. Press enter to begin: ", input.nextLine() );
        System.out.println("\nStarting offline video predictions");
        do {
        	newFrame = new Instance(CoreVars.queueSize+1);
        	for (int k = 0; k < CoreVars.queueSize; k++) newFrame.setValue((Attribute)fvWekaAttributes.elementAt(k), mags.getVal(k));
            dataUnlabeled.add(newFrame);
            dataUnlabeled.setClassIndex(dataUnlabeled.numAttributes() - 1);
            // Predict
            isDistracted = eTest.evaluateModelOnceAndRecordPrediction(cModel, dataUnlabeled.lastInstance());
        	if (isDistracted == 1) {
        		if (feed.state == 0) feed.showRed();
        	} else {
        		if (feed.state == 1 || feed.state == -1) feed.showGreen();
        	}

        } while ( readNextFrame() );
        
    }
    
    /**
     * Collect training data from a given video
     * 
     * @param vidFPS -  video file frames per second
     * @param stampT - time stamp relative to video that a classification occurs (in seconds)
     * @param vidLenSecs - video length in seconds
     * @param startT - time in seconds where code should "fast forward" to in video
     * @param vidFile - video file name with file extension
     * @param lblsFile - manual classification data file with extension
     * @throws IOException
     */
    void captureFromVidFile(double vidFPS, int stampT, double vidLenSecs, int startT, String vidFile, String outFile) throws IOException {
    	trainFile = outFile;
    	cam = new VideoCapture(vidFile);
    	System.out.printf("Fast forwarding %ds into the video\n", startT);
    	for(int i = 0; i < (int) (startT*vidFPS); i++) cam.read(frame); //fast forward to where video should begin recording
    	
    	System.out.println("DATA COLLECTION BEGINNING");
    	CoreVars.queueSize = (int) (stampT*vidFPS);
    	CoreVars.rows = (int) vidLenSecs/stampT;
    	
    	System.out.printf("vid name: %s\nQueue Size is: %d\n", vidFile, CoreVars.queueSize);
        
        //Weka  objects
    	Attribute[] attributes;
    	attributes = new Attribute[CoreVars.queueSize+1];
        fvWekaAttributes = new FastVector(4); // Declare the feature vector
        
    	for (int i = 0; i < CoreVars.queueSize; i++) {
    		attributes[i] = new Attribute(String.format("magnitude%d", i) ); // create numeric attributes
    		fvWekaAttributes.addElement(attributes[i]);
    	}
        FastVector fvClassVal = new FastVector(2); // Declare the class attribute along with its values
        fvClassVal.addElement(labels[0]); fvClassVal.addElement(labels[1]);
        attributes[CoreVars.queueSize] = new Attribute("theClass", fvClassVal);
        fvWekaAttributes.addElement(attributes[CoreVars.queueSize]);

        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, CoreVars.rows); // Create an empty training set
        // Set class index
        isTrainingSet.setClassIndex(CoreVars.queueSize);

        Instance newReading;
    	//input = new Scanner( new File(lblsFile) );
    	for (int i = 0; i < CoreVars.rows; i++ ) {
    		
    		if ( fillUpQueue() ) {
                newReading = new Instance(CoreVars.queueSize+1);
                for (int k = 0; k < CoreVars.queueSize; k++) newReading.setValue((Attribute) fvWekaAttributes.elementAt(k), mags.getVal(k));
                
                newReading.setValue((Attribute)fvWekaAttributes.elementAt(CoreVars.queueSize), labels[type] );
                isTrainingSet.add(newReading); // add the instance
    		}
    	}
        saveArff(isTrainingSet, trainFile);
    	
    }

    /**
     * Collect training data from web camera
     */
    void train() throws IOException  {
    	System.out.println("TRAINNING IS BEGINNING");
    	int i;

        //WEKA OBJECTS
    	Attribute[] attributes;
    	attributes = new Attribute[CoreVars.queueSize+1];
        fvWekaAttributes = new FastVector(4); // Declare the feature vector
        
    	for (i = 0; i < CoreVars.queueSize; i++) {
    		attributes[i] = new Attribute(String.format("magnitude%d", i) ); // create numeric attributes
    		fvWekaAttributes.addElement(attributes[i]);
    	}
        FastVector fvClassVal = new FastVector(2); // Declare the class attribute along with its values
        fvClassVal.addElement(labels[0]);
        fvClassVal.addElement(labels[1]);
        attributes[CoreVars.queueSize] = new Attribute("theClass", fvClassVal);
        fvWekaAttributes.addElement(attributes[CoreVars.queueSize]);
        // queueSize := training length; allocate equal room for attentive and inattentive
        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, CoreVars.rows*2); // Create an empty training set
        // Set class index
        isTrainingSet.setClassIndex(CoreVars.queueSize);

        Instance newReading;        
        for (i = 0; i < labels.length; i++) {
            System.out.printf("Now Collecting %s dataset. Press enter to begin: ", labels[i]);
            input.nextLine();
            
            fillUpQueue();
            
            for (int j = 0; j < CoreVars.rows; j++) {
                newReading = new Instance(CoreVars.queueSize+1);
                for (int k = 0; k < CoreVars.queueSize; k++) newReading.setValue((Attribute)fvWekaAttributes.elementAt(k), mags.getVal(k));
                
                newReading.setValue((Attribute)fvWekaAttributes.elementAt(CoreVars.queueSize), labels[i]);
                isTrainingSet.add(newReading); // add the instance
                readNextFrame();
            }
        }
        saveArff(isTrainingSet, trainFile);
    }

    /**
     * Test training data collected from webcam against web cam feed
     * @throws Exception
     */
    void test() throws Exception {

    	System.out.println("TESTING IS BEGINNING");
    	Instances isTrainingSet = loadArff(trainFile);
    	isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

    	// After training data set is complete, build model
        Classifier cModel = (Classifier)new RandomForest();   
        cModel.buildClassifier(isTrainingSet);
        Evaluation eTest = new Evaluation(isTrainingSet);
                
        // Time variables
        long cTime, sTime, secondsWait = 5l;
        boolean wasDistracted = false;
        double isDistracted;
        Instances dataUnlabeled = new Instances("TestInstances", fvWekaAttributes, 0);
        
        System.out.print("Press enter to begin: ");
        input.nextLine();
        
        Instance newFrame;
        
        fillUpQueue();
        sTime = new  Date().getTime();
        for(int i = 0; i < CoreVars.rows; i++) {
        	
        	newFrame = new Instance(CoreVars.queueSize+1);
        	for (int k = 0; k < CoreVars.queueSize; k++) newFrame.setValue((Attribute)fvWekaAttributes.elementAt(k), mags.getVal(k));
            dataUnlabeled.add(newFrame);
            dataUnlabeled.setClassIndex(dataUnlabeled.numAttributes() - 1);
            // Predict
            isDistracted = eTest.evaluateModelOnceAndRecordPrediction(cModel, dataUnlabeled.lastInstance());
            System.out.println(isDistracted);
            cTime = new Date().getTime();
           
            // Also throw in a check for whether driver has been alerted in last ~5s
            if (isDistracted == 1 && wasDistracted  && (cTime - sTime) / 1000l >= secondsWait) {
            	System.out.print( (cTime - sTime) );
            	System.out.println("\t" + labels[1]); // Turn this into an alert on the phone
            	sTime = cTime; // update most recent alert
            }   
            // Set up for next frame
            if (isDistracted == 1) wasDistracted = true;
            else wasDistracted = false;
            
            readNextFrame();
        }
    }

    /** Saves the Instances object as an .arff file.
     * Pre-conditions: location specifies an absolute path.
     * Post-conditions: saves dataSet at location.
     */
    void saveArff(Instances dataSet, String saveFile) throws IOException {
		 ArffSaver saver = new ArffSaver();
		 saver.setInstances(dataSet);
		 saver.setFile(new File(saveFile));
		 saver.writeBatch();
    }
    
    /** Loads an Instances object from an .arff file.
     * Pre-conditions: location specifies an absolute path.
     * Post-conditions: loads Instances from location.
     */
    Instances loadArff(String location) throws IOException {
    	BufferedReader reader = new BufferedReader(new FileReader(location));
    	ArffReader arff = new ArffReader(reader);
    	return arff.getData();
    }
     
}