import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.analysis.MultivariateVectorFunction
import org.apache.commons.math3.analysis.differentiation.MultivariateDifferentiableFunction
import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.optim.InitialGuess
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.OptimizationData
import org.apache.commons.math3.optim.SimpleBounds
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer
import org.apache.commons.math3.util.FastMath
import org.junit.Test

import org.junit.Assert.*
import java.util.*

class NeuralNetworkTest {

    @Test
    fun calculateGradients() {
        val Theta1 = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("Theta1.txt")).readMatrix(25, 401)
        val Theta2 = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("Theta2.txt")).readMatrix(10, 26)
        val data = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("test_data.txt")).readMatrix(5000, 401)
        //println(data)
        NeuralNetwork().apply {
            addLayer(400)
            addLayer(25)
            addLayer(10)
            initialize()
            set(0, Theta1.transpose())
            set(1, Theta2.transpose())
            val gradients = calculateGradients(data.getSubMatrix(0, 4999, 0, 399), data.getColumn(400).map { it.toInt() - 1 }.toIntArray())
            assertTrue(estimateGradient(data.getSubMatrix(0, 4999, 0, 399), data.getColumn(400).map { it.toInt() - 1 }.toIntArray(), 1, 1, 0)
                    - gradients[1].getEntry(1, 0) < 1e-3)
            assertTrue(estimateGradient(data.getSubMatrix(0, 4999, 0, 399), data.getColumn(400).map { it.toInt() - 1 }.toIntArray(), 0, 0, 0)
                    - gradients[1].getEntry(0, 0) < 1e-3)
        }
    }

    @Test
    fun trainNN() {
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
            var alpha = 0.06
            var cost = 0.0
            (0 until iterations).forEach {
                cost = calculateCost(x, y.toIntArray())
                val gradients = calculateGradients(x, y.toIntArray())
                gradients.map { it.scalarMultiply(alpha) }
                (0 until gradients.size).forEach {
                    set(it, get(it).subtract(gradients[it]))
                }
            }
            println("Final network errors : $cost")
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
            reloadModel(DoubleArray(foldModel().size) { Random().nextDouble() })
            fixInput(x, y.toIntArray())
            BOBYQAOptimizer(foldModel().size * 2 + 1).apply {
                // Optimizer tests
                optimize(MaxEval(iterations), GoalType.MINIMIZE, InitialGuess(foldModel()), SimpleBounds.unbounded(foldModel().size),
                        getCostFunction(), getGradientFunction()
                ).apply {
                    reloadModel(this.point)
                }
            }
            println("Final network errors : ${calculateCost(x, y.toIntArray())}")
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

    // This test case expect a throwable, because common minimum algorithms would require large memory
    fun largeSetTests() {
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
            data = data.getSubMatrix(0, 999, 0, 400)
            fixInput(data.getSubMatrix(0, 999, 0, 399), data.getColumn(400).map { it.toInt() - 1 }.toIntArray())
            BOBYQAOptimizer(foldModel().size * 2 + 1).apply {
                // Optimizer tests
                optimize(MaxEval(100), GoalType.MINIMIZE, InitialGuess(foldModel()), SimpleBounds.unbounded(foldModel().size),
                        getCostFunction(), getGradientFunction()
                ).apply {
                    reloadModel(this.point)
                }
            }
            println("Final network errors : ${calculateCost()}")
        }
    }

    @Test
    fun calculateCost() {
        val Theta1 = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("Theta1.txt")).readMatrix(25, 401)
        val Theta2 = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("Theta2.txt")).readMatrix(10, 26)
        val data = Scanner(NeuralNetworkTest().javaClass.getResourceAsStream("test_data.txt")).readMatrix(5000, 401)
        //println(data)
        NeuralNetwork().apply {
            addLayer(400)
            addLayer(25)
            addLayer(10)
            initialize()
            set(0, Theta1.transpose())
            set(1, Theta2.transpose())
            assertTrue(FastMath.abs(calculateCost(data.getSubMatrix(0, 4999, 0, 399), data.getColumn(400).map { it.toInt() - 1 }.toIntArray())
                    - 0.287629) < 1e-6)
            assertTrue(FastMath.abs(calculateCost(data.getSubMatrix(0, 4999, 0, 399), data.getColumn(400).map { it.toInt() - 1 }.toIntArray(), 1.0)
                    - 0.383770) < 1e-6)
        }
    }
}