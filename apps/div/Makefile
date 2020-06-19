include ../support/Makefile.inc

ifeq (${SCALAR}, true)
IS_SCALAR = -DSCALAR
endif

ITERATIONS ?= 10

$(BIN)/generator : pipeline.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(IS_SCALAR) $(filter-out %.h,$^) -Iinclude -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LIBS)

$(BIN)/%/div_pipeline.o: $(BIN)/generator
	@mkdir -p $(@D)
	$^ -g fast_div -o $(@D) -e o,h -f div_pipeline target=$*-hvx_128 ${SCHEDULING_OPTS}

$(BIN)/%/process: process.cpp $(BIN)/%/div_pipeline.o
	@mkdir -p $(@D)
	$(CXX-$*) $(CXXFLAGS) $(CXXFLAGS-$*) -Iinclude -I $(BIN)/$*/  $(DASH_D_DEFINES) -Wall -O0 -g process.cpp $(BIN)/$*/div_pipeline.o -o $(@D)/process $(LDFLAGS-$*)

test: $(BIN)/$(HL_TARGET)/process
	$< -n $(ITERATIONS)

clean:
	rm -rf $(BIN)
