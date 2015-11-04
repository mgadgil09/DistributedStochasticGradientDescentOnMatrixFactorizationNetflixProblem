import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

object dsgd1 {
  def main(args: Array[String]) {
    val logFile = "YOUR_SPARK_HOME/README.md" // Should be some file on your system
    val conf = new SparkConf().setAppName("Distributed SGD")
    val sc = new SparkContext(conf)
    val file = sc.textFile("input.txt")
//didn't know i had to work on 2000*2000 data
//now i have to create distributed matrix.
//creating ratings from my original data
var ratings = file.map{umr => org.apache.spark.mllib.recommendation.Rating(umr.split(",")(0).toInt,umr.split(",")(1).toInt, umr.split(",")(2).toInt)}
//now creating distributed co-ordinate matrix
val distMatrix = new CoordinateMatrix(ratings.map {case org.apache.spark.mllib.recommendation.Rating(user, movie, rating) => MatrixEntry(user, movie, rating)}).toBlockMatrix.toLocalMatrix
var myMatrix = new org.apache.spark.mllib.linalg.DenseMatrix(distMatrix.numRows,distMatrix.numCols,distMatrix.toArray)
var breezeMyMatrix = new breeze.linalg.DenseMatrix(distMatrix.numRows,distMatrix.numCols,distMatrix.toArray)
 var Blknum = 2
    def indexCreation(dummyrows:Array[Int],Blk:Int):Array[Array[Int]] = {
    var B = Blk
    var B1 = B
    var j=0
    var dummysize = dummyrows.length
    var dummyrowscopy = dummyrows
    var Blockrows = Array.ofDim[Int](B1,dummysize)
    	while(B>0){
    	var num = dummysize/B
 
	    for(i <- 0 to num-1){
	    Blockrows(j)(i) = dummyrowscopy.apply(i)
	
	         }
	    Blockrows(j) = Blockrows(j).slice(0,num)
	    dummyrowscopy = dummyrowscopy.slice(num,dummyrowscopy.length)
	    dummysize = dummyrowscopy.length
	    B = B-1
	    j = j+1
	       }

    return Blockrows
	}
val rowIndices = (for(i <- 0 to myMatrix.numRows-1)yield(i)).toArray   
val colIndices = (for(i <- 0 to myMatrix.numCols-1)yield(i)).toArray   
 var blockrow = indexCreation(rowIndices,Blknum)
    var blockcol = indexCreation(colIndices,Blknum)
//cartesian products of row and col indices
var car = sc.parallelize(blockrow).cartesian(sc.parallelize(blockcol))
var makeStratums = car.map{case(x,y) => stratumMaker(x,y)}
var stratumRdds = for(i <- makeStratums.collect()) yield {sc.parallelize(i)}
var stratums = stratumRdds.map{case mentry => new CoordinateMatrix(mentry).toBlockMatrix.toLocalMatrix}
//initializing W0 and H0
var F = 3
var w0 = org.apache.spark.mllib.linalg.DenseMatrix.rand(myMatrix.numRows,F,new java.util.Random)
var h0 = org.apache.spark.mllib.linalg.DenseMatrix.rand(F,myMatrix.numCols,new java.util.Random)
//breeze Vij, W0, H0

var breezeVij = stratums.map{stratum => new breeze.linalg.DenseMatrix(stratum.numRows,stratum.numCols,stratum.toArray)}
//my matrices in breeze Densematrix format

val breezew0 = new breeze.linalg.DenseMatrix[Double](w0.numRows,w0.numCols,w0.values)
val breezeh0 = new breeze.linalg.DenseMatrix[Double](h0.numRows,h0.numCols,h0.values)
//joining stratums with original matrix indices
//I have my stratums
def stratumMaker(x:Array[Int],y:Array[Int]):breeze.linalg.DenseMatrix[Double] = {
	var stratumEntries = for(j <- 0 to y.length-1)yield{
				for(i <- 0 to x.length-1) yield{
					distMatrix.apply(x(i),y(j))
						}
					  }
var stratumMatrix = new breeze.linalg.DenseMatrix[Double](x.length,y.length,stratumEntries.flatten.toArray)
	//var stratumRatings = stratumEntries.flatten.map{case(umr) => org.apache.spark.mllib.recommendation.Rating(umr.split(":")(0).toInt, umr.split(":")(1).toInt, umr.split(":")(2).toDouble.toInt)}
        //var stratumMatrix = stratumRatings.map{case org.apache.spark.mllib.recommendation.Rating(user, movie, rating) => MatrixEntry(user, movie, rating)}
stratumMatrix
      
}
def wMatrixMaker(iarr:Array[Int],jarr:Array[Int],wMatrix:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double] ={
var subWMatrixValues =   for(allcols <-0 to wMatrix.cols-1)yield{
			  for(wi <- 0 to iarr.length-1)yield{
	                   		wMatrix.apply(iarr(wi),allcols)
				}
			}
var subMatrix = new breeze.linalg.DenseMatrix[Double](iarr.length,wMatrix.cols,subWMatrixValues.flatten.toArray)
subMatrix	
}

def hMatrixMaker(iarr:Array[Int],jarr:Array[Int],hMatrix:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double] ={
var subWMatrixValues =   for(wj <-0 to jarr.length-1)yield{
			  for(allrows <- 0 to hMatrix.rows-1)yield{
	                   		hMatrix.apply(allrows,jarr(wj))
				}
			}
var subMatrix = new breeze.linalg.DenseMatrix[Double](hMatrix.rows,jarr.length,subWMatrixValues.flatten.toArray)
subMatrix	
}
 var zippedBlkRow = sc.parallelize(blockrow).zipWithIndex.map{case(k,v) => (v,k)}
 var zippedBlkCol = sc.parallelize(blockcol).zipWithIndex.map{case(k,v) => (v,k)}
 var carWithIndex = zippedBlkRow.cartesian(zippedBlkCol)
var indexedIndices = carWithIndex.map{case((i, arr1), (j, arr2)) => ((arr1,arr2),(i,j))}
var indexedStratums = sc.parallelize(breezeVij).zipWithIndex.map{case(k,v) => (v,k)}.join(car.zipWithIndex.map{case(k,v) => (v,k)}).map{case(a,(b,c)) => (c,b)}
var indices = indexedIndices.filter{case((a,b),(i,j)) => {i == 0 && j == 0}}.map{case(k,v) => k}.collect()(0)

for(x <- 0 to blockrow.size)yield{
	for(y <- 0 to blockcol.size)yield{
	   var stratsIndices = for(i <- 0 to blockrow.size-1; j <- 0 to blockcol.size-1
				if i==j)yield{
			    var xArr =  indexedIndices.filter{case((a,b),(i1,j1)) => {i1 == i.toInt && j1 == j.toInt}}.map{case((a,b),(c,d)) => a}.collect()(0)
			    var yArr =  indexedIndices.filter{case((a,b),(i1,j1)) => {i1 == i.toInt && j1 == j.toInt}}.map{case((a,b),(c,d)) => b}.collect()(0)
			    var VStratum =  stratumMaker(xArr,yArr)
			    var WStratum = wMatrixMaker(xArr,yArr,breezew0)	  
	  		    var HStratum = hMatrixMaker(xArr,yArr,breezeh0)
				(VStratum,WStratum,HStratum)
	 				   }
		

    		}
	}
             


}}
