#!/bin/bash

script_path=$(cd `dirname ${BASH_SOURCE}`;pwd -P)
app_path="${script_path}/../src"

. ${script_path}/func_util.sh


function check_param_configure()
{
    for i in `cat ${app_path}/param_configure.conf | awk -F'[ =]+' '{print $2}'`
    do
        if [[ ${i} = "" ]];then
            echo "please check your param_configure.conf to make sure that each parameter has a value"
            return 1
        fi
    done 
    
    #get and check format of remost_host ip
    check_remote_host
    if [ $? -ne 0 ];then
		return 1
    fi
}


function build_common()
{
    echo "build common lib..."
    if [ ! -d "${HOME}/ascend_ddk" ];then
        mkdir $HOME/ascend_ddk
        if [[ $? -ne 0 ]];then
            echo "ERROR: Execute mkdir command failed, Please check your environment"
            return 1
        fi
    fi
    bash ${script_path}/build_ezdvpp.sh ${remote_host}
    if [ $? -ne 0 ];then
        echo "ERROR: Failed to deploy ezdvpp"
        return 1
    fi

    bash ${script_path}/build_presenteragent.sh ${remote_host}
    if [ $? -ne 0 ];then
        echo "ERROR: Failed to deploy presenteragent"
        return 1
    fi
    return 0
}


function main()
{
    touch ${script_path}/../src/graph.config
    [ $? -ne 0 ] && (echo "ERROR:touch graph.config failed";return 1)
	
    echo "Modify param information in graph.config..."
    check_param_configure
    if [ $? -ne 0 ];then
        return 1
    fi
    
    build_common
    if [ $? -ne 0 ];then
        echo "ERROR: Failed to deploy common lib"
        return 1
    fi
    return 0
}
main


















