
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

object SimpleApp {
  def main(args: Array[String]) {
    val logFile = "YOUR_SPARK_HOME/README.md" // Should be some file on your system
    val conf = new SparkConf().setAppName("Distributed SGD")
    val sc = new SparkContext(conf)
    val file = sc.textFile("input.txt")
    val userKey = file.map{line => line.split(",")}.map{arr => (arr(0).toInt,(arr(1)+":"+arr(2)))}.sortByKey()
    val ind = userKey.reduceByKey{(a,b) => a.toString+"--"+b.toString}.sortByKey().zipWithIndex	
    val user = ind.map{case((k,v),i) => (i.toInt,v.toString())}
    val user1 = user.map{case(k,v) => (k,v.split("--"))}
    val user2 = user1.flatMap{case(k,v) => {
		for(i <- 0 to v.length-1)yield(k,v(i))}}
    val user3 = user2.map{case(k,v) => (k,v.split(":")(0)+","+v.split(":")(1))}	
    val movie1 = user3.map{case(k,v) => (v.split(",")(0).toInt,k.toString+","+v.split(",")(1))}.sortByKey()	
    val movie2 = movie1.reduceByKey{(a,b) => a.toString+"--"+b.toString}.sortByKey().zipWithIndex   
    val movie3 = movie2.map{case((k,v),i) =>(i.toInt,v.toString())}.map{case(k,v) => (k,v.split("--"))}
    val movie4 = movie3.flatMap{case(k,v) => {for(i <- 0 to v.length-1)yield(k,v(i))}}
    val umr = movie4.map{case(k,v) => (v.split(",")(0).toInt,k.toString+","+v.split(",")(1))}
   
    val muri = umr.map{case(k,v) => (v.split(",")(0).toInt,k.toString+","+v.split(",")(1))}.zipWithIndex.map{case((k,v),i) => (k.toInt,v.toString+","+i.toString)}
    val colptr = muri.reduceByKey{(a,b) => {if(a.split(",")(0).toInt < b.split(",")(0).toInt) a else b }}.sortByKey()
    
    val colptrNew = colptr.map{case(k,v) => v.split(",")(2)}.collect()
    val values = umr.map{case(k,v) => v.split(",")(1).toInt}
    val lastcolptr = values.count().toInt
    var temp = Array(lastcolptr.toInt)
    var finalcolptr = colptrNew.map{line => line.toInt}
    finalcolptr = finalcolptr.union(temp)
    var rowindices = umr.map{case(k,v) => k.toInt}.collect()
    var rows = rowindices.sorted.distinct
    var cols = umr.map{case(k,v) => v.split(",")(0).toInt}.collect().sorted.distinct
    var rowcount = umr.map{case(k,v) => k}.distinct.count().toInt
    var colcount = umr.map{case(k,v) => v.split(",")(0).toInt}.distinct.count().toInt
    var doubleValues = values.map{line => line.toDouble}.collect()
    var mat = new org.apache.spark.mllib.linalg.SparseMatrix(rowcount,colcount,finalcolptr,rowindices,doubleValues)
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
    var blockrow = indexCreation(rows,Blknum)
    var blockcol = indexCreation(cols,Blknum)
//cartesian products of row and col indices
var car = sc.parallelize(blockrow).cartesian(sc.parallelize(blockcol)).collect()

  //my matrix to RDD[Vector]
    def matrixToRDD(m: Matrix):RDD[org.apache.spark.mllib.linalg.Vector] = {
   val columns = m.toArray.grouped(m.numRows)
   val rows = columns.toSeq.transpose 
   val vectors = rows.map(row => new org.apache.spark.mllib.linalg.DenseVector(row.toArray))
   sc.parallelize(vectors)

}
val distrows = matrixToRDD(mat.toDense)
 val distmat = new RowMatrix(distrows)

//now i have to create distributed matrix.
//creating ratings from my original data
var ratings = umr.map{case(user,mr) => org.apache.spark.mllib.recommendation.Rating(user.toInt, mr.split(",")(0).toInt, mr.split(",")(1).toInt)}
//now creating distributed co-ordinate matrix
val distMatrix = new CoordinateMatrix(ratings.map {case org.apache.spark.mllib.recommendation.Rating(user, movie, rating) => MatrixEntry(user, movie, rating)})
var blkMatrix = distMatrix.toBlockMatrix
var blocks = blkMatrix.blocks

var x = car.map{case(x,y) => x(0)}	
//experiment
def stratumMaker(x:Array[Int],y:Array[Int]):scala.collection.immutable.IndexedSeq[org.apache.spark.mllib.linalg.distributed.MatrixEntry] = {
	var stratumEntries = for(i <- 0 to x.length-1) yield{
			for(j <- 0 to y.length-1)yield{
		i.toString+":"+j.toString+":"+mat.apply(x(i),y(j)).toString
						}
					  }

	var stratumRatings = stratumEntries.flatten.map{case(umr) => org.apache.spark.mllib.recommendation.Rating(umr.split(":")(0).toInt, umr.split(":")(1).toInt, umr.split(":")(2).toDouble.toInt)}
        var stratumMatrix = stratumRatings.map{case org.apache.spark.mllib.recommendation.Rating(user, movie, rating) => MatrixEntry(user, movie, rating)}
stratumMatrix
      
}
var stratums = car.map{case(x,y) => stratumMaker(x,y)}
var stratumRdds = for(i <- stratums.collect())yield{sc.parallelize(i)}
var coordMatrix = stratumRdds.map{case mentry => new CoordinateMatrix(mentry).toBlockMatrix.toLocalMatrix}
	
	return stratumMatrix
		


  }
}
