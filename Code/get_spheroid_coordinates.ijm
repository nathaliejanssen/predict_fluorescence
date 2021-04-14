run("8-bit");
run("Variance...", "radius=5");
setAutoThreshold("Otsu");
setOption("BlackBackground", false);
run("Convert to Mask");
run("Options...", "iterations=10 count=3 pad do=Close");
run("Options...", "iterations=10 count=3 pad do=Open");
run("Select Bounding Box (guess background color)");

macro "List XY Coordinates" {
     ID = getTitle();
     tabDelText = "";
     getSelectionCoordinates(x, y);
     for (i=0; i<x.length; i++){
     	 tabDelText = ID+" "+i+" "+x[i]+" "+y[i];
     	 File.append(tabDelText, "C:/Users/natha/hello/selection_output/"+"pilot.txt");
     }
}


