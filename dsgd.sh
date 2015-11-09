#!/bin/bash
sbt package
bin/spark-submit --class "dsgdApp" --master local[*] target/scala-2.10/simple-project_2.10-1.0.jar

