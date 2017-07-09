import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

object dsgdApp{
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Distributed SGD")
    val sc = new SparkContext(conf)
    val file = sc.textFile("nf_subsample.csv")
    var ratings = file.map{umr => org.apache.spark.mllib.recommendation.Rating(umr.split(",")(0).toInt,umr.split(",")(1).toInt, umr.split(",")(2).toInt)}
    val distMatrix = new CoordinateMatrix(ratings.map {case org.apache.spark.mllib.recommendation.Rating(user, movie, rating) => MatrixEntry(user, movie, rating)}).toBlockMatrix.toLocalMatrix
    var distMatBcasted = sc.broadcast(distMatrix)
    var myMatrix = new org.apache.spark.mllib.linalg.DenseMatrix(distMatrix.numRows,distMatrix.numCols,distMatrix.toArray)
    var breezeMyMatrix = new breeze.linalg.DenseMatrix(distMatrix.numRows,distMatrix.numCols,distMatrix.toArray)
    var Blknum = 10	  
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
    //initializing W0 and H0
    var F = 20
    //var arr = for(i <- 0 to 14)yield{5.0}
    var w0 = org.apache.spark.mllib.linalg.DenseMatrix.rand(myMatrix.numRows,F,new java.util.Random)
    var h0 = org.apache.spark.mllib.linalg.DenseMatrix.rand(F,myMatrix.numCols,new java.util.Random)
    var breezew0 = new breeze.linalg.DenseMatrix[Double](w0.numRows,w0.numCols,w0.values)
    var breezeh0 = new breeze.linalg.DenseMatrix[Double](h0.numRows,h0.numCols,h0.values)
    var bCastedW0 = sc.broadcast(breezew0)
    var bCastedH0 = sc.broadcast(breezeh0)
    var sumMbi = 0
    var nDash = 0
    var bcastedsumMbi = sc.broadcast(sumMbi)
    var bCastedNDash = sc.broadcast(nDash)
    def stratumMaker(x:Array[Int],y:Array[Int]):breeze.linalg.DenseMatrix[Double] = {
	var stratumEntries = for(j <- 0 to y.length-1)yield{
				for(i <- 0 to x.length-1) yield{
					distMatBcasted.value.apply(x(i),y(j))
						}
					  }
        var stratumMatrix = new breeze.linalg.DenseMatrix[Double](x.length,y.length,stratumEntries.flatten.toArray)
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
    var indexedStratums = sc.parallelize(Array(breezeMyMatrix)).zipWithIndex.map{case(k,v) => (v,k)}.join(car.zipWithIndex.map{case(k,v) => (v,k)}).map{case(a,(b,c)) => (c,b)}
    var indices = indexedIndices.filter{case((a,b),(i,j)) => {i == 0 && j == 0}}.map{case(k,v) => k}.collect()(0)
    //sgd algorithm

    def calculateNewWH(v:breeze.linalg.DenseMatrix[Double],w:breeze.linalg.DenseMatrix[Double],h:breeze.linalg.DenseMatrix[Double],i:Int,j:Int):(Int,breeze.linalg.DenseMatrix[Double],Int,breeze.linalg.DenseMatrix[Double])={
	var dense = new org.apache.spark.mllib.linalg.DenseMatrix(v.rows,v.cols,v.data)
	var nonZeroElements = dense.numNonzeros
	var n = bCastedNDash.value.toDouble + bcastedsumMbi.value.toDouble
	var beta = -0.6
	var epsilon = scala.math.pow((100.0 + n),beta)
	var N = Array(v.rows,v.cols)
	for(i <- 0 to v.rows-1; j <- 0 to v.cols-1 if v.apply(i,j)!=0){	
		var xind = i
		var yind = j
		var vValue = v.apply(xind,yind)
		var wRowStar = for(i <- 0 to w.cols-1)yield{Math.getExponent(w.apply(xind,i)).toDouble}
        	var hStarCol = for(i <- 0 to h.rows-1)yield{Math.getExponent(h.apply(i,yind)).toDouble}
		var wRowStarMatrix = new breeze.linalg.DenseMatrix(1,w.cols,wRowStar.toArray)	
        	var hStarColMatrix = new breeze.linalg.DenseMatrix(h.rows,1,hStarCol.toArray)
		var dw = (-2.0)*(vValue - (wRowStarMatrix*hStarColMatrix)).apply(0,0)*(hStarColMatrix)+((2.0*0.1)/N(0))*(wRowStarMatrix.t)
		var dh = (-2.0)*(vValue - (wRowStarMatrix*hStarColMatrix)).apply(0,0)*(wRowStarMatrix.t)+((2.0*0.1)/N(1))*(hStarColMatrix)
		var wDash = wRowStarMatrix - ((epsilon*(N(0)+N(1))*dw.t))
		var hDash = hStarColMatrix - ((epsilon*(N(0)+N(1))*dh))
		wRowStarMatrix = wDash
		hStarColMatrix = hDash
		for(i <- 0 to w.cols-1){
		w(xind,i) = Math.getExponent(wRowStarMatrix.apply(0,i)).toDouble
		}
		for(i <- 0 to h.rows-1){
			h(i,yind) = Math.getExponent(hStarColMatrix.apply(i,0)).toDouble
		}
	        }  
		return (i,w,j,h)	
		}
	def createStratumsVWH(i:Int,j:Int):(breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double],breeze.linalg.DenseMatrix[Double],Int,Int)={
		var xArr =  indexedIndices.filter{case((a,b),(i1,j1)) => {i1 == i.toInt && j1 == j.toInt}}.map{case((a,b),(c,d)) => a}.collect()(0)
		var yArr =  indexedIndices.filter{case((a,b),(i1,j1)) => {i1 == i.toInt && j1 == j.toInt}}.map{case((a,b),(c,d)) => b}.collect()(0)
		var VStratum =  stratumMaker(xArr,yArr)
		var WStratum = wMatrixMaker(xArr,yArr,bCastedW0.value)	  
		var HStratum = hMatrixMaker(xArr,yArr,bCastedH0.value)	
		return (VStratum,WStratum,HStratum,i,j)
		}
	def converter(mat:breeze.linalg.DenseMatrix[Double]):Array[Double]={
		var vd = 0.0
		var Varr = for(i <- 0 to mat.rows-1)yield{
		var Varr1 = for(j <- 0 to mat.cols-1)yield{
		if(!(Math.getExponent(mat.apply(i,j)).toDouble.isNaN)){
		vd = Math.getExponent(mat.apply(i,j)).toDouble
		}
		vd
		}
		Varr1
		}
		return Varr.flatten.toArray
	    }
    //dsgd algorithm
    var output = for(iteration <- 1 to 100)yield{
    var iter = sc.broadcast(iteration)
    //for loop for diagonals
    var strats =   for(i <- 0 to blockrow.size-1; j <- 0 to blockcol.size-1
        		 if i==j )yield{
  				var diagStrats = createStratumsVWH(i,j)
				diagStrats
     }
   //Updating global value for SGD updates
   var diagonals = sc.parallelize(strats).coalesce(Blknum).map{case(v,w,h,i,j) => calculateNewWH(v,w,h,i,j)}.map{case(i,w,j,h) => (w,h)}
   //non zeros
   var v = sc.parallelize(strats).map{case(v,w,h,i,j) => v}.collect()
   //var nonzerov = v.map{case(mat) => {for(i <- mat.rows-1)}}
   //Strata 1 SGD done
   sumMbi = sumMbi + diagonals.count().toInt
   bcastedsumMbi = sc.broadcast(sumMbi)
   nDash = nDash + 1
   bCastedNDash = sc.broadcast(nDash)
   //reconstruct W and H
   var reconstructedW = diagonals.map{case(w,h) => (1,w)}.reduceByKey((a,b) => breeze.linalg.DenseMatrix.vertcat(a,b)).map{case(k,v) => v}.collect()(0)
   var reconstructedH = diagonals.map{case(w,h) => (1,h)}.reduceByKey((a,b) => breeze.linalg.DenseMatrix.horzcat(a,b)).map{case(k,v) => v}.collect()(0)
   bCastedW0 = sc.broadcast(reconstructedW)
   bCastedH0 = sc.broadcast(reconstructedH)
   //stratification for other than diagonal blocks
   for(y<- 1 to blockcol.length-1){
      var i=0; var j=y
      var loopVal = for(i <- 0 to blockrow.length-1)yield{
	 var upperStrats = createStratumsVWH(i,j)
 	 j=j+1
	 if(j>blockcol.length-1)
	 j=0 
	 upperStrats
	 }
	 var restBlocks = sc.parallelize(loopVal).coalesce(Blknum).map{case(v,w,h,i,j) => calculateNewWH(v,w,h,i,j)}
	 sumMbi = sumMbi + restBlocks.count().toInt
	 bcastedsumMbi = sc.broadcast(sumMbi)	
	 nDash = nDash + 1
	 bCastedNDash = sc.broadcast(nDash)	
	 var otherW = restBlocks.map{case(i,w,j,h) => (i,w)}.sortByKey().map{case(i,w) => (1,w)}.reduceByKey((a,b) => breeze.linalg.DenseMatrix.vertcat(a,b)).map{case(k,v) => v}.collect()(0)
	 var otherH = restBlocks.map{case(i,w,j,h) => (j,h)}.sortByKey().map{case(j,h) => (1,h)}.reduceByKey((a,b) => breeze.linalg.DenseMatrix.horzcat(a,b)).map{case(k,v) => v}.collect()(0)
       	 bCastedW0 = sc.broadcast(otherW)
	 bCastedH0 = sc.broadcast(otherH)
	 }
	var vDash = bCastedW0.value * bCastedH0.value
	var vOriginal = new breeze.linalg.DenseMatrix(distMatBcasted.value.numRows,distMatBcasted.value.numCols,distMatBcasted.value.toArray)
	var differenceV = vOriginal - vDash
	var mllibDense = converter(differenceV)	
	var wSparse = converter(bCastedW0.value)
	var hSparse = converter(bCastedH0.value)
	var nzsl = 0.0
	var wSqrSum = 0.0
	var hSqrSum = 0.0	
	for(i <- 0 to mllibDense.length-1){nzsl = nzsl + (scala.math.pow(mllibDense(i),2))}
	for(i <- 0 to wSparse.length-1){wSqrSum = wSqrSum + (scala.math.pow(wSparse(i),2))}
	for(i <- 0 to hSparse.length-1){hSqrSum = hSqrSum + (scala.math.pow(hSparse(i),2))}
	var L2 = Math.sqrt(nzsl + 0.1*(Math.sqrt(wSqrSum) + Math.sqrt(hSqrSum)))
	vDash
	}			      
    sc.parallelize(output).coalesce(1).saveAsTextFile("q1Results1")				
    }
}
