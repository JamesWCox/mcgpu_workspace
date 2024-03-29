##############################################################################
# MCGPU Makefile
# Version 2.0
#
# This Makefile depends on GNU make.
#
########################
# Makefile Target List #
########################
#
# make [all]   : Compiles all source files and builds the main application
#				 in release mode.
# make tests   : Compiles all source files and unittesting source files and
#			     builds a release versions of the testing application.
# make dirtree : Creates all of the directories needed by the makefile to
#				 store the generated output files.
# make clean   : Deletes the object and bin directories from the project
#			     folder, removing all generated files.
#
##############################
# Makefile Defined Variables #
##############################
#
# BUILD=debug    : Builds the program in debug mode
# BUILD=profile  : Builds the program in profiling mode
# BUILD=release  : Builds the program in release mode
#
# PRECISION=single : All floating point numbers use single-precision
# PRECISION=double : All floating point numbers use double-precision
#
# SHELL=path/to/file  : The relative path to the shell executable program on
#						the current machine. This shell program will allow the
#						makefile to execute system commands
#
##############################################################################

#########################
# Environment Variables #
#########################

# The location of the shell script program that will execute scripts from
# the makefile. This is specified as per GNU Makefile conventions.
SHELL = /bin/sh

#############################
# Project Structure Details #
#############################

# The relative path to the directory containing the source files
SourceDir := src

# The relative path to the directory containing the unit tests
TestDir := test

# The relative path to the object directory
ObjDir := obj

# The relative path to the bin directory
BinDir := bin

# The path to the output directory where the built executables will be stored
AppDir := $(BinDir)

# The path to the output directory where the compiled object files will be
# built, as well as the dependency information files
BuildDir := $(ObjDir)

# Defines the modules that exist in the source directory that should be
# included by the compiler. All files within these directories that are
# valid file types for the Makefile to handle will automatically be
# compiled into object files in the build directory. Make sure to add
# any new modules to this list, or else they will not be compiled.
# NOTE: If a file is no longer being used make sure to remove it from
# the module directory, else it will be included in the build.
Modules := Applications \
		Metropolis \
		Metropolis/Utilities \
		Metropolis/ParallelSim\
		Metropolis/SerialSim

########################
# Program Output Names #
########################

# The name of the main program generated by the makefile
AppName := metrosim

# The name of the unit testing program generated by the makefile
UnitTestName := metrotest

##############################
#      Compiler Settings     #
##############################

# Defines the compiler used to compile and link the source files.
CC := nvcc

# Defines compiler-specific flags
# Some of the GNU C++ library headers included by the Google Test
# infrastructure use C++11 features (notably, variadic templates)
ifeq ($(CC),nvcc)
	CFLAGS := --std=c++11 -lcudart
else
	CFLAGS := -std=c++11
endif

# Defines the types of files that the Makefile knows how to compile
# and link. Specify the filetype by using a modulus (percent sign),
# followed by a dot and the file extension (e.g. %.java, %.txt).
FileTypes := %.cpp

# Relative search paths for Include Files. We include the root folder, the
# source folder, and the testing folder.
IncPaths := . $(SourceDir) $(TestDir)

# Compiler specific flags for the C++ compiler when generating .o files
# and when generating .d files for dependency information
ifeq ($(CC),nvcc)
	CompileFlags := -x cu -arch=sm_20 -dc
else
	CompileFlags := -c
endif

# Flags for linking metrosim with the PGI compiler.
ifeq ($(CC),nvcc)
	LinkFlags := -arch=sm_20
else
	LinkFlags := -lgomp
endif

# The debug compiler flags add debugging symbols to the executable
# This originally contained -Minfo=ccff
DebugFlags := -g

# The profile compiler flags add profiling hooks (which produce gmon.out)
# as well as debugging symbols, but optimizations are still enabled
ProfileFlags := -g -O3 -pg

# The release build compiler flags that add optimization flags and remove
# all symbol and relocation table information from the executable.
ReleaseFlags := -O3

ifeq ($(CC),nvcc)
	ReleaseFlags +=
endif

#############################
# Automated Testing Details #
#############################

# The relative path to the testing module containing the unit test source.
UnitTestDir := $(TestDir)/unittests

# The relative path to the Google Test module that contains the source
# code and libraries for the Google Test framework.
GTestDir := gtest

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTestHeaders = $(GTestDir)/include/gtest/*.h \
               $(GTestDir)/include/gtest/internal/*.h

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
GTestFlags := -I$(GTestDir)/include
ifeq ($(CC),g++)
	GTestFlags += -lpthread
endif
#GTestFlags += -pthread #-Wall -Wextra

# Builds gtest.a and gtest_main.a.
# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTestDir)/src/*.cc $(GTestDir)/src/*.h $(GTestHeaders)

###########################
# Application Definitions #
###########################

# The base define list to pass to the compiled and linked executable.
Definitions := APP_NAME=\"$(AppName)\"

# Check for the BUILD definition: debug, profile, or release. If this is is not
# set by the user, then the build will default to a release build. If the user
# specifies an option other than 'debug', 'profile', or 'release', then the
# build will default to release build.
ifeq ($(BUILD),debug)
	# "Debug" build - set compiling and linking flags
	CompileFlags += $(DebugFlags)
	LinkFlags += $(DebugFlags)
	BuildDir := $(BuildDir)/debug
	Definitions += DEBUG
else
ifeq ($(BUILD),profile)
	# "Profile" build - set compiling and linking flags
	CompileFlags += $(ProfileFlags)
	LinkFlags += $(ProfileFlags)
	BuildDir := $(BuildDir)/profile
	Definitions += PROFILE

else
	# "Release" build - set compiling and linking flags
	CompileFlags += $(ReleaseFlags)
	LinkFlags += $(ReleaseFlags)
	BuildDir := $(BuildDir)/release
	Definitions += RELEASE
endif
endif

# Check for the PRECISION definition being set to single or double. If this
# define is not set by the user, then the build will default to double
# precision. If the user specifies an option other than 'single' or 'double'
# then the build will default to double precision.
ifeq ($(PRECISION),single)
        Definitions += SINGLE_PRECISION
else
        Definitions += DOUBLE_PRECISION
endif

######################
# Internal Variables #
######################

# The main application file that contains the entry point for the application
ProgramMain := $(BuildDir)/$(SourceDir)/Program.o

# Derives the compiler flags for included search paths
Includes := $(addprefix -I, $(IncPaths))

# Derives the compiler flags for defined variables for the application
Defines := $(addprefix -D, $(Definitions))

# Derives the paths to each of the source modules
SourceModules := $(addprefix $(SourceDir)/,$(Modules))

# Creates a list of folders inside the object output directory that need
# to be created for the compiled files.
ObjFolders := $(addprefix $(BuildDir)/,$(SourceModules))
ObjFolders += $(BuildDir)/$(UnitTestDir)

# Searches through the specified Modules list for all of the valid
# files that it can find and compile. Once all of the files are
# found, they are appended with an .o and prefixed with the object
# directory path. This allows the compiled object files to be routed
# to the proper output directory.
Sources := $(filter $(FileTypes),$(wildcard $(addsuffix /*,$(SourceModules))))
Objects := $(patsubst %,$(BuildDir)/%.o,$(basename $(Sources)))

# The unit testing objects are all gathered seperately because they are
# included all at once from the testing directory and are compiled into the
# output program alongside the source objects.
UnitTestingSources := $(filter %.cpp,$(wildcard $(UnitTestDir)/*))
UnitTestingObjects := $(patsubst %,$(BuildDir)/%.o,\
		      $(basename $(UnitTestingSources)))

##############################
# Dependency Graph Functions #
##############################

# The function to create the formated dependency graph file after having the
# generated compiler file (*.d) piped in from the compiler.
format_dep = sed -n "H;$$ {g;s@.*:\(.*\)@$(basename $@).o $@: \$$\(wildcard\1\)@;p}" > $@

# The command that will generate the dependency file (*.d) output for a given
# C++ file using the C++ compiler
create_dep_cpp = $(CC) $(CFLAGS) $(CompileFlags) $(Includes) $(Defines) $<

# The command that will generate the dependency file (*.d) output for a given
# unit testing file. These files are special because they use specific flags
# that link the testing framework.
create_dep_unittest = $(CC) $(CFLAGS) $(GTestFlags) $(Includes) $(Defines) $<

##############################
# Makefile Rules and Targets #
##############################

# Specifies that these make targets are not actual files and therefore will
# not break if a similar named file exists in the directory.
.PHONY : all tests $(AppName) $(UnitTestName) dirtree clean cleanDependencyObjs

# The list of build targets that the user can specify

all : $(AppName) cleanDependencyObjs

cleanDependencyObjs:
	rm -f *.o

#tests : $(AppName) $(UnitTestName) cleanDependencyObjs

$(AppName) : $(Objects) $(ProgramMain) | dirtree
	$(CC) $^ $(CFLAGS) $(Includes) $(Defines) -o $(AppDir)/$@ $(LinkFlags)

#$(UnitTestName) : $(Objects) $(UnitTestingObjects) $(ObjDir)/gtest_main.a | dirtree
#	$(CC) $^ $(CFLAGS) $(GTestFlags) $(Includes) $(Defines) -o $(AppDir)/$@ $(LinkFlags)

dirtree :
	@mkdir -p $(ObjFolders) $(BinDir) $(ObjDir) $(AppDir) $(BuildDir)

clean :
	rm -rf $(ObjDir) $(BinDir)

################################
# Object and Source File Rules #
################################


# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.

#$(ObjDir)/gtest-all.o : $(GTEST_SRCS_) | dirtree
#	$(CC) $(CFLAGS) $(GTestFlags) -I$(GTestDir) -c \
            $(GTestDir)/src/gtest-all.cc -o $@

#$(ObjDir)/gtest_main.o : $(GTEST_SRCS_) | dirtree
#	$(CC) $(CFLAGS) $(GTestFlags) -I$(GTestDir) -c \
	  $(GTestDir)/src/gtest_main.cc -o $@

#$(ObjDir)/gtest.a : $(ObjDir)/gtest-all.o | dirtree
#	$(AR) $(ARFLAGS) $@ $^

#$(ObjDir)/gtest_main.a : $(ObjDir)/gtest-all.o $(ObjDir)/gtest_main.o | dirtree
#	$(AR) $(ARFLAGS) $@ $^

# Here are the Rules that determine how to compile a C++ source
# file into an object file. Note that we have specified a special rule for
# unit testing object files, as they use a special set of compiler flags
# in order to meet the Google Test framework requirements.

#$(BuildDir)/$(UnitTestDir)/%.o : $(UnitTestDir)/%.cpp $(GTestHeaders) | dirtree
#	$(CC) $(CFLAGS) $(CompileFlags) $(GTestFlags) $(Defines) $(Includes) $< -o $@

$(BuildDir)/%.o : %.cpp | dirtree
	$(CC) $(CFLAGS) $(CompileFlags) $(Includes) $(Defines) $< -o $@

##########################
# Dependency Build Rules #
##########################

# These rules specify how to generate the dependency graph information
# for each of the object files used in linking the final executables.
# We have a special rule for unit testing because it contains special
# compiler flags.

#$(BuildDir)/$(UnitTestDir)/%.dep : $(UnitTestDir)/%.cpp | dirtree
#	$(SHELL) -ec '$(create_dep_unittest) | $(format_dep)'

#$(BuildDir)/%.dep : %.cpp | dirtree
#	$(SHELL) -ec '$(create_dep_cpp) | $(format_dep)'

######################
# Dependency Include #
######################

# This conditional statement will attempt to include all of the dependency
# files located in the object directory. If the files exist, then their
# dependency information is loaded, and each source file checks to see if
# it needs to be recompiled. The if statements are used to make sure that
# the dependency info isn't rebuilt when the object directory is being
# cleaned or when some other target is selected that doesn't generate files.

ifneq ($(MAKECMDGOALS),dirtree)
ifneq ($(MAKECMDGOALS),clean)
-include $(Objects:.o=.dep)
-include $(ProgramMain:.o=.dep)
-include $(UnitTestingObjects:.o=.dep)
endif
endif
