g_configure_ec2_instances <- function() {
    list(
        sleep_sec_cluster_start = 200, ## how many seconds to wait for a worker EC2 instance to start up, before we can ssh in
        sleep_sec_ip_addresses = 10,   ## how many seconds to wait for a secondary IP address on a worker node to become active, after it is associated with the network interface
        cluster_size_instances = 19,   ## this number is currently limited by default limits in EC2; to increase the limit, need
                                       ### to send a request their tech support, at this link:
### https://console.aws.amazon.com/support/home?region=us-west-2#/case/create?issueType=service-limit-increase&limitType=service-code-ec2-instances
        num_ip_addresses_per_instance = 2,
        ami_id = "ami-cae763aa",       ## this is the "CERENKOV_CLUSTER5" AMI
        instance_type = "c4.2xlarge",  ## "m4.2xlarge" ## need two worker processes to fully load a m4.2xlarge instance, it appears
        security_group_settings = "sg-15cdac6d",
        subnet = "subnet-370a5e40",
        username = "ubuntu",
        network_interface_device_name = "ens3",  ## NOTE:  this is probably ubuntu-specific
        subnet_cidr_suffix = 20  ## means it's a /20
    )
}

## -------------------- EC2 specific functions ----------------------
g_run_ec2_instance <- function(p_image_id,
                               p_instance_type,
                               p_subnet_id,
                               p_security_group_id,
                               p_num_ip_addresses_per_instance) {

    if (! require(aws.ec2, quietly=TRUE)) {
        stop("package aws.ec2 is missing")
    }
    
    query <- list(Action="RunInstances",
                 ImageId=p_image_id,
                 InstanceType=p_instance_type,
                 MinCount=1,
                 MaxCount=1)

    if (p_num_ip_addresses_per_instance > 1) {
        query <- c(query,
                   list(NetworkInterface.1.DeviceIndex=0,
                        NetworkInterface.1.SecurityGroupId=p_security_group_id,
                        NetworkInterface.1.SecondaryPrivateIpAddressCount=(p_num_ip_addresses_per_instance-1),
                        NetworkInterface.1.SubnetId=p_subnet_id))
    } else {
        query <- c(query,
                   list(SubnetId=p_subnet_id),
                   list(SecurityGroupId.0=p_security_group_id))
    }
    
    r <- ec2HTTP(query)
    return(lapply(r$instancesSet, `class<-`, "ec2_instance"))
}

g_get_and_configure_ip_addresses_for_ec2_instances <- function(p_ec2_instances,
                                                               p_ec2_username,
                                                               p_ec2_subnet_cidr_suffix,
                                                               p_ec2_network_interface_device_name) {
    setNames(unlist(lapply(p_ec2_instances, function(p_ec2_instance) {
        instance_ip_address_set <- p_ec2_instance$networkInterfaceSet[[1]]$privateIpAddressesSet
        primary_private_ip_address <- instance_ip_address_set[[1]]$privateIpAddress
        ret_ip_address <- primary_private_ip_address
        
        if (length(instance_ip_address_set) > 1) {
            ## get a list of all secondary IP addresses for this EC2 instance
            ret_ip_address <- c(ret_ip_address,
                                setNames(
                                    unlist(
                                        lapply(instance_ip_address_set[2:length(instance_ip_address_set)],
                                               function(p_ip_address_item) {
                                                   ## need to turn on the secondary IP address; this is almost certainly Ubuntu-specific
                                                   secondary_ip_address <- p_ip_address_item$privateIpAddress[[1]]
                                                   system_cmd <- sprintf("ssh -oStrictHostKeyChecking=no %s@%s sudo ip addr add %s/%d dev %s",
                                                                         p_ec2_username,
                                                                         primary_private_ip_address,
                                                                         secondary_ip_address,
                                                                         p_ec2_subnet_cidr_suffix,
                                                                         p_ec2_network_interface_device_name)
                                                   print(sprintf("running system command: %s", system_cmd))
                                                   system(system_cmd)
                                                   secondary_ip_address
                                               })),
                                    NULL))
        }

        ret_ip_address
    })), NULL)
}

g_create_ec2_instances <- function(p_ec2_par,
                                   p_run_ec2_instance,
                                   p_get_and_configure_ip_addresses_for_ec2_instances) {
    if (! require(aws.ec2, quietly=TRUE)) { stop("missing required package aws.ec2") }

    ## here is where we create the EC2 instances that we will use
    ec2_instances <- do.call(c, lapply(1:p_ec2_par$cluster_size_instances, function(instance_number) {
        p_run_ec2_instance(p_ec2_par$ami_id,
                           p_ec2_par$instance_type,
                           p_ec2_par$subnet,   
                           p_ec2_par$security_group_settings,
                           p_ec2_par$num_ip_addresses_per_instance)
    }))

    print(sprintf("waiting for %d seconds for cluster to start", p_ec2_par$sleep_sec_cluster_start))
    pbsapply(1:p_ec2_par$sleep_sec_cluster_start, function(x) { Sys.sleep(1) })

    ip_addresses <- p_get_and_configure_ip_addresses_for_ec2_instances(ec2_instances,
                                                                         p_ec2_par$username,
                                                                         p_ec2_par$subnet_cidr_suffix,
                                                                         p_ec2_par$network_interface_device_name)
        
    print("IP addresses for the cluster: ")
    print(ip_addresses)

    ## if we have configured secondary private IP addresses for the instances, it seems sensible to wait a few seconds for the IP addresses to become active
    if (p_ec2_par$num_ip_addresses_per_instance  > 1) {
        print(sprintf("waiting for %d seconds for secondary IP addresses to become active", p_ec2_par$sleep_sec_ip_addresses))
        pbsapply(1:p_ec2_par$sleep_sec_ip_addresses, function(x) { Sys.sleep(1) })
    }

    print(sprintf("creating the PSOCK cluster"))

    cluster <- makeCluster(ip_addresses,
                           type="SOCK",
                           outfile="/dev/null",
                           rshcmd="ssh -oStrictHostKeyChecking=no",
                           useXDR=TRUE,
                           methods=FALSE)

    terminator_func <- function() {
        stopCluster(cluster)
        terminate_instances(ec2_instances)
    }
    
    list(cluster=cluster,
         ec2_instances=ec2_instances,
         ip_addresses=ip_addresses,
         terminator_func=terminateor_func)
} 
    
## :TODO: switch this function to use the AWS REST API rather than the AWS CLI
g_make_message_notifier_function <- function(p_aws_sns_topic_arn) {
    if (! is.null(p_aws_sns_topic_arn)) {
        function(p_message_text) {
            ## using ignore.stderr to suppress some deprecation warning from the AWS CLI on macOS
            system(paste("aws sns publish --topic-arn \"",
                         p_aws_sns_topic_arn,
                         "\" --message \"",
                         p_message_text,
                         "\"",
                         sep=""),
                   ignore.stdout=TRUE,
                   ignore.stderr=TRUE)
            print(p_message_text)
        }
    } else {
        print
    }    
}

