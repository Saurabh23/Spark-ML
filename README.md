#SPARK off the  *Million Song DataSet*

Once you clone the project do:

1. from within the root folder run "sbt"
2. from within sbt run "update"
3. from within sbt run "eclipse"

The sbt eclipse command makes sure your project can now be correctly imported into eclipse IDE.

All spark logs are going into the spark.log.

##Creating JAR:

1. open file build.sbt
2. In the value of `libraryDependencies` change "compile" to "provided"
3. specify the main class in value of `mainClass in assembly`
4. run `sbt assembly` 
  NOTE: for compiling the file in IntelliJ revert "provided" to "compile" in `libraryDependencies`
