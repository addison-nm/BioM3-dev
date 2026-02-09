### Runtime concerns

* Datatype issues for save/load 16bit vs 32 bit precision. When request 32 bit prec in config file, training runs. Upon check after training, the loaded test model ends up being 16 bit, throwing an assertion error. Current fix is to explicitly request 16 bit in config.


