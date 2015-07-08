
import org.opencv.core.Point;

import java.util.ArrayList;

public class Records {
	ArrayList<Point> data;
	ArrayList<Double> databug;
	private int expLength;
	
	public Records(int expL) {
		data = new ArrayList<Point>();
		databug = new ArrayList<Double>();
		expLength = expL;
	}
	
	public void add (Point pt) {
		if (data.size()+1 > expLength)
			data.remove(0);
		data.add(pt);	
	}
	
	public void add(double mag) {
		if (databug.size()+1 > expLength)
			databug.remove(0);
		databug.add(mag);
	}

}
