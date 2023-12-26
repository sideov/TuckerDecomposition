import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.rdd.RDD


// val data = Array(
//   MatrixEntry(0,0,1.0),
//   MatrixEntry(0,1,1.0),
//   MatrixEntry(0,2,1.0)
//   )
val data = Array(
  MyTensorEntry(Array(0, 0, 0), 1),
  MyTensorEntry(Array(0, 0, 1), 2),
  MyTensorEntry(Array(0, 0, 2), 3),
  MyTensorEntry(Array(0, 1, 0), 4),
  MyTensorEntry(Array(0, 1, 1), 5),
  MyTensorEntry(Array(0, 1, 2), 6),
  MyTensorEntry(Array(0, 2, 0), 7),
  MyTensorEntry(Array(0, 2, 1), 8),
  MyTensorEntry(Array(0, 2, 2), 9),

  MyTensorEntry(Array(1, 0, 0), 10),
  MyTensorEntry(Array(1, 0, 1), 11),
  MyTensorEntry(Array(1, 0, 2), 12),
  MyTensorEntry(Array(1, 1, 0), 13),
  MyTensorEntry(Array(1, 1, 1), 14),
  MyTensorEntry(Array(1, 1, 2), 15),
  MyTensorEntry(Array(1, 2, 0), 16),
  MyTensorEntry(Array(1, 2, 1), 17),
  MyTensorEntry(Array(1, 2, 2), 18),

  MyTensorEntry(Array(2, 0, 0), 19),
  MyTensorEntry(Array(2, 0, 1), 20),
  MyTensorEntry(Array(2, 0, 2), 21),
  MyTensorEntry(Array(2, 1, 0), 22),
  MyTensorEntry(Array(2, 1, 1), 23),
  MyTensorEntry(Array(2, 1, 2), 24),
  MyTensorEntry(Array(2, 2, 0), 25),
  MyTensorEntry(Array(2, 2, 1), 26),
  MyTensorEntry(Array(2, 2, 2), 27)
)
val data2 = Array (
  MatrixEntry(0,0,1),
  MatrixEntry(0,1,2),
  MatrixEntry(0,2,3),
  MatrixEntry(1,0,4),
  MatrixEntry(1,1,5),
  MatrixEntry(1,2,6),
  MatrixEntry(2,0,7),
  MatrixEntry(2,1,8),
  MatrixEntry(2,2,9)
)
val data3 = Array (
  MyTensorEntry(Array(0, 0, 0), 1),
  MyTensorEntry(Array(0, 1, 0), 4),
  MyTensorEntry(Array(0, 2, 0), 7),
  MyTensorEntry(Array(0, 3, 0), 10),
  MyTensorEntry(Array(1, 0, 0), 2),
  MyTensorEntry(Array(1, 1, 0), 5),
  MyTensorEntry(Array(1, 2, 0), 8),
  MyTensorEntry(Array(1, 3, 0), 11),
  MyTensorEntry(Array(2, 0, 0), 3),
  MyTensorEntry(Array(2, 1, 0), 6),
  MyTensorEntry(Array(2, 2, 0), 9),
  MyTensorEntry(Array(2, 3, 0), 12),

  MyTensorEntry(Array(0, 0, 1), 13),
  MyTensorEntry(Array(0, 1, 1), 16),
  MyTensorEntry(Array(0, 2, 1), 19),
  MyTensorEntry(Array(0, 3, 1), 22),
  MyTensorEntry(Array(1, 0, 1), 14),
  MyTensorEntry(Array(1, 1, 1), 17),
  MyTensorEntry(Array(1, 2, 1), 20),
  MyTensorEntry(Array(1, 3, 1), 23),
  MyTensorEntry(Array(2, 0, 1), 15),
  MyTensorEntry(Array(2, 1, 1), 18),
  MyTensorEntry(Array(2, 2, 1), 21),
  MyTensorEntry(Array(2, 3, 1), 24),
)

def printMatrixEntries(entries: Array[MatrixEntry]): Unit = {
  val numRows = entries.map(_.i.toInt).max + 1
  val numCols = entries.map(_.j.toInt).max + 1

  val matrix = Array.ofDim[Double](numRows, numCols)
  entries.foreach(entry => matrix(entry.i.toInt)(entry.j.toInt) = entry.value)

  for (i <- 0 until numRows) {
    for (j <- 0 until numCols) {
      print(s"${matrix(i)(j)}\t")
    }
    println()
  }
}

val rows = spark.sparkContext.parallelize(data3)
val matrisatedTensor : RDD[MatrixEntry] = MyTensorEntryMatrisation(rows, mode=2)

val collectedEntries: Array[MatrixEntry] = matrisatedTensor.collect()

// collectedEntries.foreach(println)
printMatrixEntries(collectedEntries)
// 

val mat: CoordinateMatrix = new CoordinateMatrix(matrisatedTensor)
/* Хотим создать CoordinateMatrix из RDD поверх MyTensorEntry

Для этого надо реализвать функцию
*/
val rowmat: RowMatrix = mat.toRowMatrix()

case class MyTensorEntry(coords: Array[Long], value: Double)

def MyTensorEntryMatrisation(tensor: RDD[MyTensorEntry], mode: Integer) : RDD[MatrixEntry] = {

  val flatMapResult = tensor.flatMap(entry => entry.coords.zipWithIndex)
  val flatMapResult2 = flatMapResult.map(entry=>(entry._2, entry._1))
  val groupByKeyResult = flatMapResult2.reduceByKey((x,y)=>math.max(x,y))
  val sortByKeyResult = groupByKeyResult.sortByKey()
  val iss = sortByKeyResult.map(entry=>entry._2+1)
  val is = iss.collect()

  println(is.mkString(", "))

  def calculateJs(Is: Array[Int], n: Int): Array[Int] = {
    val N = Is.length

    val Jk = new Array[Int](N+1)
    Jk(1) = 1
    if (n == 0) {
      Jk(2) = 1
    } else {
      Jk(2) = Is(0)
    }

    for (m <- 3 to N) {
      if (m == n+2) {
        Jk(m) = Jk(m - 1)
      } else {
        Jk(m) = Jk(m-1) * Is(m-2)
      }
    }

    Jk
  }
  val Js = calculateJs(is.map(_.toInt), mode)
  println(Js.mkString(", "))


  def calculatej(entry: MyTensorEntry, mode: Int, js: Array[Int]): Int = {
    (1 to is.length)
      .filter(k => k != mode+1)
      .map(k => (entry.coords(k-1)) * js(k))
      .map(k => k.toInt)
      .sum
  }

  println(calculatej(MyTensorEntry(Array(0,2,1),1), 1, Js))

  tensor.map(entry => {
    val i = entry.coords(mode)
    val Fi = calculatej(entry, mode, Js)
    // MatrixEntry(i, Fi.toLong, entry.value)
    MatrixEntry(i,Fi,entry.value)
  })

}

/*
3. Реализуем HOSVD (сразу после реализовации функции MyTensorEntryMatrisation). Для начала получить окржуения ядра. Если будет время, реализовать умножение по моде => сразу получим ядро.
4. — Реализовать HOOI - снова, если реализовано умножение по моде.

до 29 и после 3
*/


// Compute the top 5 singular values and corresponding singular vectors.
val svd: SingularValueDecomposition[RowMatrix, Matrix] = rowmat.computeSVD(1, computeU = true)
val U: RowMatrix = svd.U  // The U factor is a RowMatrix.
val s: Vector = svd.s     // The singular values are stored in a local dense vector.
val V: Matrix = svd.V     // The V factor is a local dense matrix.



def HOSVD(x: RDD[MyTensorEntry]) = {
  val As = (0 until x.first().coords.length).map { n =>

    val matrisatedTensor : RDD[MatrixEntry] = MyTensorEntryMatrisation(rows, mode=n)
    val mat: CoordinateMatrix = new CoordinateMatrix(matrisatedTensor)
    val rowmat: RowMatrix = mat.toRowMatrix()

    val svd = rowmat.computeSVD(1, computeU=true)

    svd.U

  }.toArray
}

val As = HOSVD(rows)
println(As)



