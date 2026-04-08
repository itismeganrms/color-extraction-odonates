//import qupath.ext.djl.DjlZoo
//println ai.djl.engine.Engine.getEngine("PyTorch")
//DjlZoo.logAvailableModels()
System.properties["ai.djl.offline"] = "false"
println ai.djl.engine.Engine.getEngine("PyTorch")