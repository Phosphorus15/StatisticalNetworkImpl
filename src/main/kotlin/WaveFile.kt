import org.apache.commons.math3.ml.clustering.Clusterable
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.apache.commons.math3.ml.clustering.MultiKMeansPlusPlusClusterer
import java.io.File
import java.io.PrintStream

class Point(val x: Double, val y: Double) : Clusterable {

    override fun getPoint() = arrayOf(x, y).toDoubleArray()

    override fun toString(): String {
        return "($x, $y)"
    }

}

fun main(args: Array<String>) {
    var data = File("D:\\violin.txt").readLines().map { it.toDouble() }.mapIndexed { index, i -> Point(index.toDouble(), i) }
    val out = PrintStream("D:\\violin_seg.txt")
    data = data.subList(20000, 21000)
    var alternate = 0
    MultiKMeansPlusPlusClusterer<Point>(KMeansPlusPlusClusterer<Point>(6), 1000).apply {
        cluster(data).forEach {
            println("${it.center.point[0]}, ${it.center.point[1]}")
            it.points.forEach {
                out.println("${it.x},${it.y},$alternate")
            }
            alternate ++
        }
    }
}
