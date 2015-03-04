package com.jingchen.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.jingchen.ann.DataNode;

public class DataUtil
{
	private static DataUtil instance = null;
	private Map<String, Integer> mTypes;
	private int mTypeCount;
	private DataUtil()
	{
		mTypes = new HashMap<String, Integer>();
		mTypeCount = 0;
	}
	public static synchronized DataUtil getInstance()
	{
		if(instance == null)
			instance = new DataUtil();
		return instance;
		
	}
	public Map<String, Integer> getTypeMap()
	{
		return mTypes;
	}
	public int getTypeCount(){
		return mTypeCount;
	}
	public String getTypeName(int type){
		if(type == -1)
			return new String("无法判断");
		Iterator<String> keys = mTypes.keySet().iterator();
		while(keys.hasNext()){
			String key = keys.next();
			if(mTypes.get(key) == type)
				return key;
		}
		return null;
	}
	/**
	 * 根据文件生成训练集
	 * @param fileName
	 * @return
	 * @throws Exception
	 */
	public List<DataNode> getDataList(String fileName, String sep) throws Exception{
		List<DataNode> list = new ArrayList<DataNode>();
		BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));
		String line = null;
		while((line = br.readLine()) != null){
			String splits[] = line.split(sep);
			DataNode node = new DataNode();
			int i = 0;
			for(; i < splits.length-1; i++)
				node.addAttrib(Float.valueOf(splits[i]));
			if(!mTypes.containsKey(splits[i]))
			{
				mTypes.put(splits[i], mTypeCount);
				mTypeCount++;
			}
			node.setType(mTypes.get(splits[i]));
			list.add(node);
		}
		return list;
	}
}
