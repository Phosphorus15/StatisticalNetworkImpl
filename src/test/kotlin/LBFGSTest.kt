import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.optim.InitialGuess
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.SimpleBounds
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer
import org.junit.Assert.*
import org.junit.Test
import org.riso.numerical.LBFGS
import java.io.PrintStream
import java.util.*

class LBFGSTest {

    @Test
    fun usability() {
        val lbfgs = LBFGS()
        val theta = DoubleArray(1) { Random().nextDouble() }
        val print = arrayOf(0, 3).toIntArray()
        val flag = IntArray(1)
        val diag = DoubleArray(theta.size)
        var it = 0
        do {
            lbfgs.lbfgs(1, 3, theta, theta[0] * theta[0] + theta[0] - 3,
                    DoubleArray(1) { 2 * theta[0] + 1 }, false, diag, print, 1e-9, 1e-10, flag)
            it++
        } while (flag[0] != 0)
        assertEquals(theta[0], .5, 1e-9)
    }

    @Test
    fun lbfgsIterator() {
        val iterations = 50000
        val x = MatrixUtils.createRealMatrix(4, 2)
        val y = arrayOf(0, 1, 1, 0)
        x.setRow(0, arrayOf(1.0, 1.0).toDoubleArray())
        x.setRow(1, arrayOf(1.0, 0.0).toDoubleArray())
        x.setRow(2, arrayOf(0.0, 1.0).toDoubleArray())
        x.setRow(3, arrayOf(0.0, 0.0).toDoubleArray())
        NeuralNetwork().apply {
            addLayer(2)
            addLayer(2)
            addLayer(3)
            addLayer(2)
            initialize()
            fixInput(x, y.toIntArray())
            val lambda = 0.000001
            var it = 0
            var theta = DoubleArray(0)
            trainNetwork(lambda).apply {
                while (hasNext() && (it++ <= 150)) {
                    theta = next()
                }
            }
            reloadModel(theta)
            println(calculateCost(inputX, inputY, lambda))
            println("Examining Model (XOR gate)")
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(1.0, 1.0).toDoubleArray())).text())
            print("(1,1) - ")
            println(predict(arrayOf(1.0, 1.0).toDoubleArray()))
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(0.0, 1.0).toDoubleArray())).text())
            print("(0,1) - ")
            println(predict(arrayOf(0.0, 1.0).toDoubleArray()))
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(1.0, 0.0).toDoubleArray())).text())
            print("(1,0) - ")
            println(predict(arrayOf(1.0, 0.0).toDoubleArray()))
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(0.0, 0.0).toDoubleArray())).text())
            print("(0,0) - ")
            println(predict(arrayOf(0.0, 0.0).toDoubleArray()))
        }
    }

    @Test
    fun basicUsage() {
        val iterations = 50000
        val x = MatrixUtils.createRealMatrix(4, 2)
        val y = arrayOf(0, 1, 1, 0)
        x.setRow(0, arrayOf(1.0, 1.0).toDoubleArray())
        x.setRow(1, arrayOf(1.0, 0.0).toDoubleArray())
        x.setRow(2, arrayOf(0.0, 1.0).toDoubleArray())
        x.setRow(3, arrayOf(0.0, 0.0).toDoubleArray())
        NeuralNetwork().apply {
            addLayer(2)
            addLayer(2)
            addLayer(3)
            addLayer(2)
            initialize()
            fixInput(x, y.toIntArray())
            val lbfgs = LBFGS()
            val theta = foldModel()
            val print = arrayOf(0, 3).toIntArray()
            val flag = IntArray(1)
            val diag = DoubleArray(theta.size)
            var it = 0
            val lambda: Double = 0.00001
            do {
                //println("${it++} ${flag[0]}")
                lbfgs.lbfgs(theta.size, 3, theta, calculateCost(inputX, inputY, lambda),
                        calculateGradients(inputX, inputY, lambda).fold(), false, diag, print, 1e-9, 0.0, flag)
                reloadModel(theta)
            } while (flag[0] != 0)
            println(calculateCost(inputX, inputY, lambda))
            println("Examining Model (XOR gate)")
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(1.0, 1.0).toDoubleArray())).text())
            print("(1,1) - ")
            println(predict(arrayOf(1.0, 1.0).toDoubleArray()))
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(0.0, 1.0).toDoubleArray())).text())
            print("(0,1) - ")
            println(predict(arrayOf(0.0, 1.0).toDoubleArray()))
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(1.0, 0.0).toDoubleArray())).text())
            print("(1,0) - ")
            println(predict(arrayOf(1.0, 0.0).toDoubleArray()))
            println(predictResult(MatrixUtils.createRowRealMatrix(arrayOf(0.0, 0.0).toDoubleArray())).text())
            print("(0,0) - ")
            println(predict(arrayOf(0.0, 0.0).toDoubleArray()))
        }
    }

    @Test
    fun largeData() {
        val Theta1 = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("Theta1.txt")).readMatrix(25, 401)
        val Theta2 = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("Theta2.txt")).readMatrix(10, 26)
        var data = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("test_data.txt")).readMatrix(5000, 401)
        //println(data)
        NeuralNetwork().apply {
            println(Runtime.getRuntime().freeMemory() / 1024 / 1024)
            addLayer(400)
            addLayer(25)
            addLayer(10)
            initialize()
            println("starting ...")
            println(Runtime.getRuntime().freeMemory() / 1024 / 1024)
            data = data.getSubMatrix(0, 4999, 0, 400)
            fixInput(data.getSubMatrix(0, 4999, 0, 399), data.getColumn(400).map { it.toInt() - 1 }.toIntArray())
            val lbfgs = LBFGS()
            val theta = foldModel()
            val print = arrayOf(-1, 3).toIntArray()
            val flag = IntArray(1)
            val diag = DoubleArray(theta.size)
            var it = 0
            val lambda: Double = 1.0
            do {
                println("${it++} ${flag[0]}")
                lbfgs.lbfgs(theta.size, 3, theta, calculateCost(inputX, inputY, lambda),
                        calculateGradients(inputX, inputY, lambda).fold(), false, diag, print, 1e-9, 0.0, flag)
                reloadModel(theta)
                if (flag[0] == -1) System.err.println("Error Detected")
                if (it >= 150) break
            } while (flag[0] != 0)
            println(calculateCost(inputX, inputY, lambda))
            System.setOut(PrintStream("D:\\output.txt"))
            println(predictResult(data.getSubMatrix(0, 4999, 0, 399)))
        }
    }
}