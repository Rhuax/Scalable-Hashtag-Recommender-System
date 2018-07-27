package UtilityClass

object Timer {

  def time[R](block: => R): (R, Double) = {
    val t0 = System.currentTimeMillis()
    val result = block    // call-by-name
    val t1 = System.currentTimeMillis()
    (result, t1 - t0)
  }

}