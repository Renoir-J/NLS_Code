% Yueyun Chen
% 20241219 v1 basic VISA communication package

classdef VISA_IO_v1 < handle

    properties (SetAccess = protected)
        address {mustBeTextScalar} = ""; % default instrument address
        default_delay {mustBeNonnegative} = 0; % default communication delay
        default_opc {mustBeNumericOrLogical} = true; % query *OPC? or not
    end

    methods (Access = public)

        % constructor
        function obj = VISA_IO_v1(addr,test)
            arguments
                addr {mustBeTextScalar}
                test {mustBeNumericOrLogical} = true; % test connection or not
            end
            obj.address = addr;
            
            % test connection
            if test
                test_session = obj.connect();
                id = obj.query(test_session,"*IDN?");
                disp("Passed connection test: " + id)
                clear test_session
            end
        end

        % update instrument address
        function update_address(obj,addr)
            arguments
                obj 
                addr {mustBeTextScalar} 
            end
            obj.address = addr;
        end
        
        % update default communication delay
        function update_delay(obj,delay)
            arguments
                obj 
                delay {mustBeNonnegative}
            end
            obj.default_delay = delay;
        end

        % update default OPC query
        function update_opc(obj,opc)
            arguments
                obj 
                opc {mustBeNumericOrLogical}
            end
            obj.default_opc = opc;
        end
    
        % send individual command, not recommended for mass communication
        function writeline(obj,message,delay)
            arguments
                obj 
                message {mustBeTextScalar} % command to send
                delay {mustBeNonnegative} = obj.default_delay; % delay time
            end
            dev = obj.connect(); % initiate connection
            obj.send(dev,message,delay,false)
            clear dev % close the connection
        end
        
        % query individual command, not recommended for mass communication
        function out = writeread(obj,message,delay)
            arguments
                obj 
                message {mustBeTextScalar} % command to send
                delay {mustBeNonnegative} = obj.default_delay; % delay time
            end
            dev = obj.connect(); % initiate connection
            out = obj.query(dev,message,delay);
            clear dev % close the connection
        end
        
        % read binary, not recommended for mass communication
        function out = read(obj,count,datatype,delay)
            arguments
                obj 
                count {mustBeNumeric} % number of values to read
                datatype {mustBeText} % data type to interpret the readings
                delay {mustBeNonnegative} = obj.default_delay; % delay time
            end
            dev = obj.connect(); % initiate connection
            out = obj.read_raw(dev,count,datatype,delay);
            clear dev % close the connection
        end

    end

    methods (Access = protected)

        % open and return the resource
        function dev = connect(obj,delay)
            arguments
                obj 
                delay {mustBeNonnegative} = obj.default_delay; % delay time
            end

            dev = visadev(obj.address); % open a session
            configureTerminator(dev,"LF") % configure line terminator
            pause(delay) % wait for finishing
        end

        % combine an array of commands
        function output = combine_commands(obj,commands)
            arguments
                obj 
                commands {mustBeText} % a string array of commands
            end
            
            output = join(commands,newline);
        end

        % query command with delay, no initiation or closing resources
        function out = query(obj,dev,message,delay)
            arguments
                obj 
                dev {mustBeA(dev,"visalib.Resource")} % visa session
                message {mustBeTextScalar} % command to send
                delay {mustBeNonnegative} = obj.default_delay; % delay time
            end
            out = dev.writeread(message); % send command
            pause(delay) % wait for finishing
        end

        % send command with delay and OPC query, no initiation or closing resources
        function send(obj,dev,message,delay,opc)
            arguments
                obj 
                dev {mustBeA(dev,"visalib.Resource")} % visa session
                message {mustBeTextScalar} % command to send
                delay {mustBeNonnegative} = obj.default_delay; % delay time
                opc {mustBeNumericOrLogical} = obj.default_opc; % query *OPC? or not
            end

            if opc
                message = obj.combine_commands([message,"*OPC?"]);
            end
            
            dev.writeline(message) % send command
            pause(delay) % wait for finishing
            
            if opc
                tempRead = dev.readline();
                pause(delay)
            end
        end

        % read raw with delay, no initiation or closing resources
        function out = read_raw(obj,dev,count,datatype,delay)
            arguments
                obj 
                dev {mustBeA(dev,"visalib.Resource")} % visa session
                count {mustBeInteger} % number of values to read
                datatype {mustBeTextScalar} % datatype to read
                delay {mustBeNonnegative} = obj.default_delay; % delay time
            end
            out = dev.read(count,datatype); % send command
            pause(delay) % wait for finishing
        end

        % write raw with delay, no initiation or closing resources
        function write_raw(obj,dev,data,datatype,delay)
            arguments
                obj 
                dev {mustBeA(dev,"visalib.Resource")} % visa session
                data % binary data to write
                datatype {mustBeTextScalar} % datatype to write
                delay {mustBeNonnegative} = obj.default_delay; % delay time
            end
            dev.write(data,datatype); % send command
            pause(delay) % wait for finishing
        end

    end

end