import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.analysis.MultivariateVectorFunction
import org.apache.commons.math3.analysis.function.Sigmoid
import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealMatrixChangingVisitor
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient
import org.apache.commons.math3.util.FastMath
import org.riso.numerical.LBFGS
import java.util.*

fun randomDoubles(size: Int) = Random().doubles().limit(size.toLong()).toArray()!!

fun RealMatrix.padding() = MatrixUtils.createRealMatrix(rowDimension, columnDimension + 1).apply {
    (1 until columnDimension).forEach {
        setColumn(it, this@padding.getColumn(it - 1))
    }
    setColumn(0, DoubleArray(rowDimension) { 1.0 })
}!!

fun Scanner.readMatrix(rows: Int, columns: Int) = MatrixUtils.createRealMatrix(rows, columns).apply {
    (0 until rows).forEach { i ->
        (0 until columns).forEach { j ->
            //println("$i $j")
            if (this@readMatrix.hasNextDouble())
                setEntry(i, j, this@readMatrix.nextDouble())
            else this@readMatrix.next()
        }
    }
}!!

fun List<RealMatrix>.fold(): DoubleArray {
    val list = mutableListOf<Double>()
    forEach {
        (0 until it.rowDimension).forEach { i ->
            (0 until it.columnDimension).forEach { j ->
                list += it.getEntry(i, j)
            }
        }
    }
    return list.toDoubleArray()
}

fun RealMatrix.multiplyEach(matrix: RealMatrix) =
        (0 until rowDimension).forEach { i ->
            (0 until columnDimension).forEach { j ->
                multiplyEntry(i, j, matrix.getEntry(i, j))
            }
        }

fun RealMatrix.text() = toString().let { it -> it.substring(it.indexOf('{') + 1) }

fun main(args: Array<String>) {
    // 'nor' network
    NeuralNetwork().apply {
        addLayer(2)
        addLayer(1)
        addLayer(2)
        initialize()
        val x = MatrixUtils.createRealMatrix(4, 2)
        val y = arrayOf(1, 0, 0, 1)
        x.setRow(0, arrayOf(1.0, 1.0).toDoubleArray())
        x.setRow(1, arrayOf(1.0, 0.0).toDoubleArray())
        x.setRow(2, arrayOf(0.0, 1.0).toDoubleArray())
        x.setRow(3, arrayOf(0.0, 0.0).toDoubleArray())
        println(calculateCost(x, y.toIntArray()))
        println(calculateGradients(x, y.toIntArray()))
    }

}

class FunctionVisitor(private val func: (Double) -> Double) : RealMatrixChangingVisitor {

    override fun start(p0: Int, p1: Int, p2: Int, p3: Int, p4: Int, p5: Int) {}

    override fun end() = 0.0

    override fun visit(p0: Int, p1: Int, p2: Double) = func(p2)

}

private class Gradient(private val network: NeuralNetwork) : MultivariateVectorFunction {
    override fun value(point: DoubleArray?): DoubleArray = network.apply { reloadModel(point!!) }.calculateGradients().fold()
}

private class Function(private val network: NeuralNetwork) : MultivariateFunction {
    override fun value(point: DoubleArray?): Double = network.apply { reloadModel(point!!) }.calculateCost()
}

class NeuralNetwork {

    private val layers: MutableList<RealMatrix> = mutableListOf()

    var input: Int = 0

    var output: Int = 0

    var consolidated = false

    var inputX: RealMatrix = MatrixUtils.createRealMatrix(1, 1)

    var inputY: RealMatrix = MatrixUtils.createRealMatrix(1, 1)

    private val sigmoid = Sigmoid()

    private fun getLayer(index: Int) = layers[index]

    operator fun get(index: Int) = getLayer(index)
    operator fun set(index: Int, matrix: RealMatrix) {
        assert(matrix.columnDimension == layers[index].columnDimension && matrix.rowDimension == layers[index].rowDimension); layers[index] = matrix
    }

    fun addLayer(size: Int) {
        if (consolidated) throw RuntimeException("This has been a fixed Neural Network")
        if (input == 0)
            input = size
        else {
            layers.add(MatrixUtils.createRealMatrix(output + 1, size).apply {
                (0 until columnDimension).forEach { i ->
                    setColumn(i, randomDoubles(rowDimension))
                }
                //println("" + this.rowDimension + " , " + this.columnDimension)
            })
        }
        output = size
    }

    fun initialize() {
        if (input == 0) throw RuntimeException("Empty Network - Could not be initialized")
        consolidated = true
    }

    fun predictResult(x: RealMatrix): RealMatrix {
        if (x.columnDimension != input) throw RuntimeException("Illegal input size")
        var inter = x.copy()!!
        //println(layers.size)
        for (layer in layers) {
            inter = inter.padding()
            //println("${inter.rowDimension} - ${inter.columnDimension}")
            inter = inter.multiply(layer)
            inter.walkInColumnOrder(FunctionVisitor(sigmoid::value))  // run sigmoid function
        }
        return inter
    }

    fun predict(x: DoubleArray): Int =
            predictResult(MatrixUtils.createRowRealMatrix(x)).getRow(0).mapIndexed { index: Int, d: Double -> Pair(index, d) }
                    .maxWith(kotlin.Comparator { o1, o2 -> o1.second.compareTo(o2.second) })!!.first

    fun fixInput(x: RealMatrix, y: IntArray) = fixInput(x,
            MatrixUtils.createRealMatrix(y.size, output).apply {
                y.forEachIndexed { index, i ->
                    setEntry(index, i, 1.0)
                }
            }
    )

    fun calculateGradients(x: RealMatrix, y: IntArray, lambda: Double = 0.0) = calculateGradients(x,
            MatrixUtils.createRealMatrix(y.size, output).apply {
                y.forEachIndexed { index, i ->
                    setEntry(index, i, 1.0)
                }
            }, lambda
    )

    fun calculateCost(x: RealMatrix, y: IntArray, lambda: Double = 0.0) = calculateCost(x,
            MatrixUtils.createRealMatrix(y.size, output).apply {
                y.forEachIndexed { index, i ->
                    setEntry(index, i, 1.0)
                }
            }, lambda
    )

    fun calculateGradients(x: RealMatrix = inputX, y: RealMatrix = inputY, lambda: Double = 0.0): List<RealMatrix> {
        val layersNeuron: MutableList<RealMatrix> = mutableListOf()
        val layersDelta: MutableList<RealMatrix> = mutableListOf()
        val gradientDelta: MutableList<RealMatrix> = mutableListOf()
        if (x.columnDimension != input) throw RuntimeException("Illegal input size")
        var inter = x.copy()!!
        for (layer in layers) { // forward propagation - vectorized
            inter = inter.padding()
            gradientDelta.add(MatrixUtils.createRealMatrix(layer.rowDimension, layer.columnDimension))
            //println("${inter.rowDimension} - ${inter.columnDimension}")
            inter = inter.multiply(layer)
            inter.walkInColumnOrder(FunctionVisitor(sigmoid::value))  // run sigmoid function
            layersNeuron.add(inter.copy())
        }
        layersDelta.add(0, inter.subtract(y))
        layersNeuron.add(0, x)
        //println(layersNeuron.size)
        val it = layersNeuron.reversed().iterator()
        it.next()
        for (layer in layers.reversed()) { // backward propagation
            //println("layer $layer")
            //println(layersDelta[0])
            val delta = layer.multiply(layersDelta[0].transpose()).transpose()
            var temp = it.next().copy()
            //println(temp)
            val reverse = temp.copy()
            reverse.walkInColumnOrder(FunctionVisitor { 1.0 })
            temp.multiplyEach(reverse.subtract(temp))
            temp = temp.padding()
            //println("t delta $delta $temp")
            delta.multiplyEach(temp)
            layersDelta.add(0, delta.getSubMatrix(0, delta.rowDimension - 1, 1, delta.columnDimension - 1))
        }
        //println("thetas")
        //layers.forEach { println(it) }
        //println("As")
        //layersNeuron.forEach { println(it) }
        //println("deltas")
        //layersDelta.forEach { println(it) }
        //return
        //println(gradientDelta.size)
        (0 until x.rowDimension).forEach { i ->
            (0 until gradientDelta.size).forEach { j ->
                val d = layersDelta[j + 1].getRowMatrix(i).transpose().multiply(layersNeuron[j].getRowMatrix(i).padding()) // delta(j)(i, :)' * [1 a(j)(i, :)]
                gradientDelta[j] = gradientDelta[j].add(d.transpose())
            }
        }
        return gradientDelta.mapIndexed { index, matrix ->
            matrix.add(layers[index].copy().apply { setRow(0, DoubleArray(columnDimension) { 0.0 }) }.scalarMultiply(lambda))
        }.map {
            it.scalarMultiply(1 / x.rowDimension.toDouble())
        }
    }

    fun calculateCost(x: RealMatrix = inputX, y: RealMatrix = inputY, lambda: Double = 0.0): Double {
        val predict = predictResult(x)
        //println(predict)
        val reverse = predict.copy()
        val m = x.rowDimension
        reverse.walkInColumnOrder(FunctionVisitor { 1.0 })
        val left = predict.copy()
        var right = predict.copy()
        left.walkInColumnOrder(FunctionVisitor(FastMath::log))
        right = reverse.subtract(right)
        right.walkInColumnOrder(FunctionVisitor(FastMath::log))
        left.multiplyEach(y)
        right.multiplyEach(reverse.subtract(y))
        var suml = 0.0
        var sumr = 0.0
        var thetaSum = 0.0
        left.walkInColumnOrder(FunctionVisitor { suml += it; it })
        right.walkInColumnOrder(FunctionVisitor { sumr += it; it })
        val baseCost = -((suml / m) + (sumr / m))
        layers.forEach {
            //println(it)
            it.walkInRowOrder(FunctionVisitor { thetaSum += it * it; it }, 1, it.rowDimension - 1, 0, it.columnDimension - 1)
        }
        thetaSum /= m
        thetaSum *= (.5) * lambda
        return thetaSum + baseCost
    }

    fun estimateGradient(x: RealMatrix, y: IntArray, layer: Int, row: Int, column: Int, epsilon: Double = 1e-9): Double {
        val layerObj = layers[layer]
        val origin = layerObj.getEntry(row, column)
        layerObj.setEntry(row, column, origin + epsilon)
        val l = calculateCost(x, y)
        layerObj.setEntry(row, column, origin - epsilon)
        val r = calculateCost(x, y)
        layerObj.setEntry(row, column, origin)
        return (l - r) / (2 * epsilon)
    }

    fun foldModel(): DoubleArray = layers.fold()

    fun reloadModel(arr: DoubleArray) {
        var calcBase = 0
        layers.forEach {
            (0 until it.rowDimension).forEach { i ->
                (0 until it.columnDimension).forEach { j ->
                    it.setEntry(i, j, arr[calcBase++])
                }
            }
        }
    }

    private fun fixInput(x: RealMatrix, y: RealMatrix) {
        inputX = x.copy()
        inputY = y.copy()
    }

    fun getGradientFunction() = ObjectiveFunctionGradient(Gradient(this))

    fun getCostFunction() = ObjectiveFunction(Function(this))

    class LBFGSIterator(private val network: NeuralNetwork, private val lbfgs: LBFGS, private val lambda: Double = 0.0) : Iterator<DoubleArray> {

        private val theta = network.foldModel()
        private val print = arrayOf(-1, 3).toIntArray()
        private val flag = IntArray(1)
        private val diagnose = DoubleArray(theta.size)
        var iterations = 0

        override fun hasNext(): Boolean = flag[0] == 1 || iterations == 0

        override fun next(): DoubleArray {
            iterations ++
            network.reloadModel(theta)
            lbfgs.lbfgs(theta.size, 3, theta, network.calculateCost(network.inputX, network.inputY, lambda)
                    , network.calculateGradients(network.inputX, network.inputY, lambda).fold(), false, diagnose, print, 1e-9, 0.0, flag)
            return theta
        }

    }

    fun trainNetwork(lambda: Double = 0.0): Iterator<DoubleArray> = LBFGSIterator(this, LBFGS(), lambda)
}