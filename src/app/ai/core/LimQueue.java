package app.ai.core;

import java.util.ArrayList;

public class LimQueue<T> {
	private ArrayList<T> data;
	private int limit; //length list must maintain. 
	
	public LimQueue(int expL) {
		data = new ArrayList<T>();
		limit = expL;
	}

	public void changeLim(int lim) { limit = lim; }
		
	public void add(T mag) { //pops up any old point to make room before adding new point
		if (data.size() >= limit)
			data.remove(0);
		data.add(mag);
	}
	
	public ArrayList<T> getLst() { return data; }
	public T getVal(int i) { return data.get(i); }
	public int size() { return data.size(); }
	
	public boolean hasRoom() { return (data.size() >= CoreVars.queueSize) ? false : true; }

}