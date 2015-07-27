package app.ai.stats;

import java.util.ArrayList;

public class Stats {
	
    //private static final String faceTAG = "Stats::Stats";
	
	public Stats(){ };
	
	/** Computes the average on a  given dataset
	 * @param data - data to compute average of
	 * @return - averge
	 */
	public double getMean(ArrayList<Double> data) {
		double sum = 0.0;
		for (double num : data) sum += num;
		return sum/data.size();
	}
	
	/** Calculates the standard derivation from a given dataset and meu
	 * @param data - dataset to compute standard derivation for
	 * @param meu - mean already pre-computed from the passed data
	 * @return sample standard derivation
	 */
	public double getSTD(ArrayList<Double> data, double meu) {
		double var = 0.0;
		for (double num :data) var += Math.pow(num - meu, 2.0);
		var  /= data.size()-1;
		return Math.sqrt(var);
	}
	
	/** Computes Method of Moments of a given dataset
	 * @param data
	 * @param meu
	 * @return Method of Moments value of a given dataset for MLEs
	 */
	public double getMoments(ArrayList<Double> data, double meu) {
		double est = 0.0;
		for (double num :data)
			est += Math.pow(num / meu - 1, 2.0);
		est /= data.size();
		
		return 1 / ( est*data.size() / (data.size()-1) );
	}
	
	/** Returns a random sample subset with no replacements
	 * @param data - dataset to sample from
	 * @param len - specific size to sample from the dataset
	 * @return random sample of size len
	 */
	public ArrayList<Double> dataSample(ArrayList<Double> data, int len) {
		ArrayList<Double> subSample = new ArrayList<Double>(len);
		for(int i = 0; i < len; i++) {
			int dex = (int)( Math.random()*data.size() );
			subSample.add( data.get( dex ) );
		}
		
		return subSample;
	}
	
	/** Generates a list of 'Moments' by continuously random sampling a dataset 
	 * @param data - dataset to randomly  sample from
	 * @param picks - number of moments wanted
	 * @param samplePercent - percentage that dictates the size of the subsamples
	 * @return - picks size of moments
	 */
	public ArrayList<Double> genRandMoments(ArrayList<Double> data, int picks, double samplePercent) {
		ArrayList<Double> moments = new ArrayList<Double>(picks);
		int sampleSize = (int) (samplePercent * data.size());
		double meu = getMean(data);
		
		for (int i = 0; i < picks; i++)
			moments.add( getMoments( dataSample(data, sampleSize), meu ) );
		
		return moments;
	}
	
	/** Determines the frequencies for each possible value the passed data
	 * @param data - dataset of values to get frequencies from
	 * @param scalar - used for evaluating floating points as Doubles
	 * 	- possible values (1, 10, 100, 1000, ...)
	 * @return list of frequencies determined from the dataset and its scalar
	 */
	public int[] getFrequencies(ArrayList<Double> data, int scalar) {
		int max = 0;
		//locate the maximum value in the list
		for (double num : data) if( (int) (num*scalar) > max) max = (int) (num*scalar);
		
		int[] frequencies = new int[max];
		for (double num : data) frequencies[(int) (num*scalar)]++;
		
		return frequencies;
	}
	
	public double getMode(ArrayList<Double> data, int scalar) {
		int min = 100000, max = 0, i = 0;
		double dex = 0;
		//locate min max
		for (double num : data) { //maximum value of raw data
			if ( (int) num*scalar > max) max = (int) (num*scalar);
			if ( (int) num*scalar < min) min = (int) (num*scalar);
		}
		//calculate frequencies from data values
		int[] frequencies = new int[max];
		for (double num : data) frequencies[(int) (num*scalar)]++;
		//locate data value with highest frequency
		max = 0;
		for (i = 0; i < frequencies.length; i++)//max = highest frequency
			if (frequencies[i] > max) {
				max = frequencies[i];
				dex = i;
			}
		//set data value to original scale
		return (dex+min) / scalar;
	}
	
	public double length(double x1, double y1, double x2, double y2) { return (int) Math.sqrt( Math.pow(x1-x2, 2.0) + Math.pow(y1-y2, 2.0) ); }
	
}
