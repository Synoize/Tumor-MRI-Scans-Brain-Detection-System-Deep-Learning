import ChatBot from "../components/ChatBot";
import { useAppContext } from "../context/AppContext";
import OnboardingScreen from "./OnboardingScreen";

const Home = () => {
    const {user} = useAppContext();

    return (
        <div className="">
            {user ? <OnboardingScreen/> : <OnboardingScreen/>}
        </div>
    )
}

export default Home;
